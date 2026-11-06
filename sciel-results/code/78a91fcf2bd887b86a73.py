import time
import sys
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # Precompute all 4-cycles and their edge indices
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append((edge_idx[a][b], edge_idx[b][c], edge_idx[c][d], edge_idx[d][a]))
        # Cycle 2: a-b-d-c-a
        cycles.append((edge_idx[a][b], edge_idx[b][d], edge_idx[d][c], edge_idx[c][a]))
        # Cycle 3: a-c-b-d-a
        cycles.append((edge_idx[a][c], edge_idx[c][b], edge_idx[b][d], edge_idx[d][a]))

    # Map edge index to list of cycle indices
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)

    def get_seed_colors():
        colors = [0] * NUM_EDGES
        color_id = 0
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        process_partition(list(range(N)))
        return colors

    current_colors = np.array(get_seed_colors(), dtype=np.int32)
    best_score = len(set(current_colors))
    best_colors = current_colors.copy()
    
    # Track frequency
    freq = np.zeros(MAX_COLORS, dtype=np.int32)
    for c in current_colors:
        freq[c] += 1
        
    used_colors_mask = freq > 0
    current_score = int(np.sum(used_colors_mask))
    
    random.seed(42)
    np.random.seed(42)
    
    temperature = 2.0
    cooling_rate = 0.9999
    min_temperature = 0.05
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
    def is_valid_change(e, new_color):
        # Check all cycles containing edge e
        for c_idx in edge_to_cycles[e]:
            c = cycles[c_idx]
            # c is a tuple of 4 edge indices
            # We need to check if the set of colors has size 4
            # To avoid creating a set every time, we can do manual checks
            # But set creation for 4 items is quite fast in Python
            # Let's optimize slightly:
            c0 = new_color if c[0] == e else current_colors[c[0]]
            c1 = new_color if c[1] == e else current_colors[c[1]]
            c2 = new_color if c[2] == e else current_colors[c[2]]
            c3 = new_color if c[3] == e else current_colors[c[3]]
            
            # Manual distinct check
            if c0 != c1 and c0 != c2 and c0 != c3 and \
               c1 != c2 and c1 != c3 and \
               c2 != c3:
                return False
        return True

    # Main Loop
    while time.time() < deadline:
        steps += 1
        
        # Strategy: 70% try to introduce new color, 30% random SA move
        if random.random() < 0.70 and current_score < MAX_COLORS:
            # Get unused colors
            unused_indices = np.where(freq == 0)[0]
            if len(unused_indices) == 0:
                # No unused colors, fallback to random
                new_color = random.randint(0, MAX_COLORS - 1)
                e = random.randint(0, NUM_EDGES - 1)
            else:
                new_color = int(random.choice(unused_indices))
                e = random.randint(0, NUM_EDGES - 1)
        else:
            # Standard SA move: pick random edge and color
            # Bias towards colors that are currently used (to avoid wasting moves on unused if we don't gain score)
            # Or bias towards rare colors to potentially consolidate and free up space?
            # Let's just pick random edge and random color for simplicity in this branch
            e = random.randint(0, NUM_EDGES - 1)
            new_color = random.randint(0, MAX_COLORS - 1)
            
        old_color = current_colors[e]
        
        if new_color == old_color:
            continue
            
        if is_valid_change(e, new_color):
            # Calculate score change
            # If old_color frequency was 1, it disappears (-1)
            # If new_color frequency was 0, it appears (+1)
            old_freq = freq[old_color]
            new_freq = freq[new_color]
            
            score_change = 0
            if old_freq == 1:
                score_change -= 1
            if new_freq == 0:
                score_change += 1
                
            new_score = current_score + score_change
            
            # Acceptance
            if new_score >= current_score:
                accept = True
            else:
                delta = current_score - new_score
                # Only accept worsening moves if temperature is high enough
                accept = random.random() < np.exp(-delta / temperature)
                
            if accept:
                # Update state
                freq[old_color] -= 1
                freq[new_color] += 1
                current_colors[e] = new_color
                
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_colors = current_colors.copy()
                    no_improve_steps = 0
                    temperature = 2.0
                else:
                    no_improve_steps += 1
                    
                temperature *= cooling_rate
                if temperature < min_temperature:
                    temperature = min_temperature
                    
                # Diversification
                if no_improve_steps > max_no_improve:
                    # Restart from best
                    current_colors = best_colors.copy()
                    freq = np.zeros(MAX_COLORS, dtype=np.int32)
                    for c in current_colors:
                        freq[c] += 1
                    current_score = best_score
                    temperature = 2.0
                    no_improve_steps = 0
                    
                    # Perturb
                    for _ in range(20):
                        pe = random.randint(0, NUM_EDGES - 1)
                        pc = random.randint(0, MAX_COLORS - 1)
                        if pc != current_colors[pe] and is_valid_change(pe, pc):
                            old_pc = current_colors[pe]
                            freq[old_pc] -= 1
                            freq[pc] += 1
                            current_colors[pe] = pc
                            # Update score
                            if freq[old_pc] == 0: current_score -= 1
                            if freq[pc] == 1: current_score += 1
                            break

    return best_colors.tolist()
