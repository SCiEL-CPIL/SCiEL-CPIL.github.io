import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    # Precompute all K4s and their edge indices
    k4s = []
    edge_index = {}
    edge_ij = []  # map edge index -> (i, j)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            edge_ij.append((i, j))
            idx += 1
            
    for a,b,c,d in combinations(range(n), 4):
        edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        edge_idxs = [edge_index[e] for e in edges]
        k4s.append(edge_idxs)
        
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edge_idxs in enumerate(k4s):
        for ei in edge_idxs:
            edge_to_k4s[ei].append(k4_idx)
            
    # Initial: (i+j) % 17
    colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    def check_edge_change(colors, edge_idx, new_color):
        """Check if changing colors[edge_idx] to new_color maintains validity."""
        old_color = colors[edge_idx]
        if old_color == new_color:
            return True
        
        # Only check K4s containing this edge
        for k4_idx in edge_to_k4s[edge_idx]:
            edge_idxs = k4s[k4_idx]
            distinct_colors = set()
            for ei in edge_idxs:
                if ei == edge_idx:
                    distinct_colors.add(new_color)
                else:
                    distinct_colors.add(colors[ei])
            if len(distinct_colors) < 5:
                return False
        return True

    def normalize(colors):
        """Relabel colors to 0..k-1"""
        used = sorted(set(colors))
        mapping = {old: new for new, old in enumerate(used)}
        return [mapping[c] for c in colors], len(used)

    current_colors = colors[:]
    best_colors = current_colors[:]
    best_num_colors = len(set(current_colors))
    
    random.seed(42)
    
    while time.time() < deadline:
        # Normalize to keep color range small
        current_colors, current_num = normalize(current_colors)
        if current_num < best_num_colors:
            best_num_colors = current_num
            best_colors = current_colors[:]
            
        if best_num_colors <= 13: # Theoretical lower bound
            break

        # Phase 1: Random Walk
        # Try to change a random edge to a random color
        for _ in range(50):
            if time.time() > deadline:
                break
            edge_idx = random.randint(0, 119)
            new_color = random.randint(0, current_num) # Try introducing new color occasionally?
            # Let's stick to existing colors + 1 to avoid explosion
            if check_edge_change(current_colors, edge_idx, new_color):
                current_colors[edge_idx] = new_color
        
        # Phase 2: Attempt Elimination
        used_colors = sorted(set(current_colors))
        if len(used_colors) <= 1:
            continue
            
        # Pick a color to eliminate
        # Heuristic: pick the one with fewest edges
        color_counts = {c: 0 for c in used_colors}
        for c in current_colors:
            color_counts[c] += 1
            
        target_color = min(used_colors, key=lambda c: color_counts[c])
        
        # Snapshot
        snapshot = current_colors[:]
        
        # Try to eliminate target_color
        edges_with_target = [i for i in range(120) if current_colors[i] == target_color]
        
        # Order edges: maybe try those with fewer options first?
        # For now, just iterate
        eliminated = True
        for edge_idx in edges_with_target:
            # Try to change this edge to any other color
            found_move = False
            # Try existing colors first, then new ones
            candidates = [c for c in used_colors if c != target_color]
            # Also try a new color (current_num)
            candidates.append(current_num)
            
            random.shuffle(candidates)
            
            for new_color in candidates:
                if check_edge_change(current_colors, edge_idx, new_color):
                    current_colors[edge_idx] = new_color
                    found_move = True
                    break
            
            if not found_move:
                eliminated = False
                break
        
        if eliminated:
            # Success! Update best
            current_colors, new_num = normalize(current_colors)
            if new_num < best_num_colors:
                best_num_colors = new_num
                best_colors = current_colors[:]
        else:
            # Failure, revert
            current_colors = snapshot
            
    # Final validation and return
    # Ensure best_colors is valid
    valid = True
    for edge_idxs in k4s:
        if len(set(best_colors[ei] for ei in edge_idxs)) < 5:
            valid = False
            break
            
    if not valid:
        # Fallback to initial
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    return best_colors
