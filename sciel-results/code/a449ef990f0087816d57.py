import numpy as np
import random
import time
import itertools

V = 16
N_SUBSETS = 1 << V
ALL_TRIPLES = list(itertools.combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES) # 560

# Precompute pair masks for linearity check
# Pair is (i, j) with i < j. Map to index i*16 + j or similar.
PAIR_INDEX = {}
pair_list = []
for i in range(V):
    for j in range(i+1, V):
        pair_list.append((i, j))
        PAIR_INDEX[(i, j)] = len(pair_list) - 1

# For each triple, get the 3 pair indices
TRIPLE_PAIRS = []
for t in ALL_TRIPLES:
    pairs = []
    for i in range(3):
        for j in range(i+1, 3):
            a, b = t[i], t[j]
            if a > b: a, b = b, a
            pairs.append(PAIR_INDEX[(a, b)])
    TRIPLE_PAIRS.append(pairs)

# Precompute supersets for each triple (NumPy boolean arrays)
# triple_supersets[i] is a boolean array of size 2^16
# triple_supersets[i][s] is True if subset s contains triple i
TRIPLE_SUPERSETS = []
subset_masks = np.arange(N_SUBSETS, dtype=np.int32)
for t in ALL_TRIPLES:
    mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    sup = (subset_masks & mask) == mask
    TRIPLE_SUPERSETS.append(sup)

SUBSET_SIZES = np.array([bin(i).count('1') for i in range(N_SUBSETS)], dtype=np.int8)

def compute_alpha(edge_indices):
    if not edge_indices:
        return V
    ind = np.ones(N_SUBSETS, dtype=bool)
    for idx in edge_indices:
        ind &= ~TRIPLE_SUPERSETS[idx]
    return int(np.max(SUBSET_SIZES[ind]))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Greedy Start
    best_alpha = V
    best_edges = []
    
    for _ in range(20): # Multiple greedy restarts
        edges = []
        used_pairs = set()
        shuffled = ALL_TRIPLES[:]
        random.shuffle(shuffled)
        
        for t in shuffled:
            idx = ALL_TRIPLES.index(t) # This is slow, use a map
            # Better: iterate by index
        # Let's use indices directly
        edges_idx = []
        used_pairs = set()
        shuffled_idx = list(range(N_TRIPLES))
        random.shuffle(shuffled_idx)
        
        for idx in shuffled_idx:
            t = ALL_TRIPLES[idx]
            pairs = TRIPLE_PAIRS[idx]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                edges_idx.append(idx)
                for p in pairs:
                    used_pairs.add(p)
        
        alpha = compute_alpha(edges_idx)
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = edges_idx[:]
            
        if time.time() > deadline - 100:
            break

    # Local Search
    current_edges = best_edges[:]
    current_alpha = best_alpha
    best_alpha = current_alpha
    best_edges = current_edges[:]
    
    used_pairs = set()
    for idx in current_edges:
        for p in TRIPLE_PAIRS[idx]:
            used_pairs.add(p)
            
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    all_indices = list(range(N_TRIPLES))
    
    while time.time() < deadline:
        move_type = random.random()
        move_applied = False
        undo_info = None
        
        if move_type < 0.4:
            # Add
            idx = random.choice(all_indices)
            if idx in current_edges: continue # Optimization: check membership? Set is better.
            # Use a set for current edges for O(1) lookup? 
            # current_edges is a list. Converting to set each time is expensive.
            # Let's maintain a set `current_edge_set` alongside list.
            # Or just check pairs. If pairs are used, it's invalid.
            pairs = TRIPLE_PAIRS[idx]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                current_edges.append(idx)
                for p in pairs:
                    used_pairs.add(p)
                move_applied = True
                undo_info = ('add', idx)
                
        elif move_type < 0.7:
            # Remove
            if not current_edges: continue
            pos = random.randint(0, len(current_edges)-1)
            idx = current_edges[pos]
            current_edges.pop(pos)
            for p in TRIPLE_PAIRS[idx]:
                used_pairs.remove(p)
            move_applied = True
            undo_info = ('remove', idx, pos)
            
        else:
            # Swap
            if not current_edges: continue
            pos = random.randint(0, len(current_edges)-1)
            idx_old = current_edges[pos]
            # Remove old
            current_edges.pop(pos)
            for p in TRIPLE_PAIRS[idx_old]:
                used_pairs.remove(p)
            
            # Try add new
            idx_new = random.choice(all_indices)
            pairs_new = TRIPLE_PAIRS[idx_new]
            valid = True
            for p in pairs_new:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                current_edges.append(idx_new)
                for p in pairs_new:
                    used_pairs.add(p)
                move_applied = True
                undo_info = ('swap', idx_old, pos, idx_new)
            else:
                # Undo remove
                current_edges.insert(pos, idx_old)
                for p in TRIPLE_PAIRS[idx_old]:
                    used_pairs.add(p)
                continue
        
        if not move_applied:
            continue
            
        new_alpha = compute_alpha(current_edges)
        
        if new_alpha < current_alpha:
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_edges = current_edges[:]
        elif new_alpha == current_alpha:
            if random.random() < 0.5:
                pass
            else:
                # Undo
                if undo_info[0] == 'add':
                    current_edges.pop()
                    for p in TRIPLE_PAIRS[undo_info[1]]:
                        used_pairs.remove(p)
                elif undo_info[0] == 'remove':
                    current_edges.insert(undo_info[2], undo_info[1])
                    for p in TRIPLE_PAIRS[undo_info[1]]:
                        used_pairs.add(p)
                elif undo_info[0] == 'swap':
                    current_edges.pop()
                    for p in TRIPLE_PAIRS[undo_info[3]]:
                        used_pairs.remove(p)
                    current_edges.insert(undo_info[2], undo_info[1])
                    for p in TRIPLE_PAIRS[undo_info[1]]:
                        used_pairs.add(p)
        else:
            delta = new_alpha - current_alpha
            if T > 0:
                prob = np.exp(-delta / T)
            else:
                prob = 0
            if random.random() < prob:
                current_alpha = new_alpha
            else:
                # Undo
                if undo_info[0] == 'add':
                    current_edges.pop()
                    for p in TRIPLE_PAIRS[undo_info[1]]:
                        used_pairs.remove(p)
                elif undo_info[0] == 'remove':
                    current_edges.insert(undo_info[2], undo_info[1])
                    for p in TRIPLE_PAIRS[undo_info[1]]:
                        used_pairs.add(p)
                elif undo_info[0] == 'swap':
                    current_edges.pop()
                    for p in TRIPLE_PAIRS[undo_info[3]]:
                        used_pairs.remove(p)
                    current_edges.insert(undo_info[2], undo_info[1])
                    for p in TRIPLE_PAIRS[undo_info[1]]:
                        used_pairs.add(p)
        T = max(T_min, T * cooling)

    # Return list of triples [a,b,c]
    return [ALL_TRIPLES[i] for i in best_edges]

# Run
if __name__ == "__main__":
    result = construct()
    # Validate
    pairs = set()
    for e in result:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = e[i], e[j]
                if a > b: a, b = b, a
                if (a,b) in pairs:
                    print("Invalid: repeated pair", (a,b))
                pairs.add((a,b))
    print("Alpha:", compute_alpha([ALL_TRIPLES.index(e) for e in result]))
