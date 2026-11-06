"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import Counter

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START

def _edge_index(i, j):
    """Map (i,j) with 0<=i<j<N to linear index in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_edge_list():
    """Return list of (i,j) in lexicographic order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _precompute_cycles():
    """
    Returns a list of all 4-cycles in K_N.
    Each cycle is a tuple of 4 edge indices.
    """
    edges = _build_edge_list()
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    def get_idx(u, v):
        if u > v: u, v = v, u
        return edge_to_idx[(u, v)]
        
    cycles = []
    
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: (a,b), (b,c), (c,d), (d,a)
                    c1 = (get_idx(a, b), get_idx(b, c), get_idx(c, d), get_idx(d, a))
                    cycles.append(c1)
                    
                    # Cycle 2: (a,b), (b,d), (d,c), (c,a)
                    c2 = (get_idx(a, b), get_idx(b, d), get_idx(d, c), get_idx(c, a))
                    cycles.append(c2)
                    
                    # Cycle 3: (a,c), (c,b), (b,d), (d,a)
                    c3 = (get_idx(a, c), get_idx(c, b), get_idx(b, d), get_idx(d, a))
                    cycles.append(c3)
                        
    return cycles

ALL_CYCLES = _precompute_cycles()
NUM_CYCLES = len(ALL_CYCLES)

# Precompute for each cycle, the 3 pairs of edges
CYCLE_PAIRS = []
for cyc in ALL_CYCLES:
    pairs = [
        (cyc[0], cyc[1]),
        (cyc[2], cyc[3]),
        (cyc[0], cyc[2]),
        (cyc[1], cyc[3]),
        (cyc[0], cyc[3]),
        (cyc[1], cyc[2])
    ]
    # Wait, a 4-cycle has 4 edges. The pairs of edges in the cycle are 6.
    # But we only need to consider merging 2 edges.
    # Any pair of edges in the cycle can be merged.
    CYCLE_PAIRS.append(pairs)

def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def _union(parent, x, y):
    rx = _find(parent, x)
    ry = _find(parent, y)
    if rx == ry:
        return False
    parent[rx] = ry
    return True

def _is_bad_cycle(cyc, parent):
    roots = set()
    for e in cyc:
        roots.add(_find(parent, e))
    return len(roots) == 4

def _greedy_merge(seed):
    random.seed(seed)
    parent = list(range(NUM_EDGES))
    
    # We will maintain a list of bad cycles
    # Initially all are bad
    bad_cycles = list(range(NUM_CYCLES))
    
    while bad_cycles:
        # Pick a random bad cycle
        idx = random.choice(bad_cycles)
        cyc = ALL_CYCLES[idx]
        
        # Check if it's still bad (might have been fixed by previous merges if we don't update perfectly, 
        # but we re-scan all, so bad_cycles is accurate at start of loop)
        # Actually, we just picked it from bad_cycles, so it should be bad.
        
        # Determine which pair to merge
        # Count how many times each pair appears in the current bad_cycles
        pair_counts = Counter()
        for b_idx in bad_cycles:
            b_cyc = ALL_CYCLES[b_idx]
            # We only care if the pair is a subset of b_cyc
            # To be fast, we can precompute a set of pairs for each cycle?
            # Or just check membership.
            # Since bad_cycles can be large, and we do this every iteration, it might be slow.
            # Optimization: 
            # Instead of counting over ALL bad_cycles, just look at the current cycle's edges.
            # Heuristic: Pick the pair of edges in `cyc` that are in the most *other* bad cycles.
            # Approximation: Just pick randomly among the 6 pairs, or the one with highest degree in the "conflict graph".
            
            # Let's use a simpler heuristic:
            # For the 6 pairs in `cyc`, count how many edges in the pair are "central" in the bad cycle set.
            # Or just random.
            pass
        
        # Let's refine the heuristic to be faster.
        # We will just pick the pair that minimizes the "loss" of future merging options?
        # No, let's stick to: Count occurrences in a sample of bad cycles if bad_cycles is large, 
        # or all if small.
        
        # Given NUM_CYCLES is ~5500, and we might have ~5000 bad cycles initially,
        # iterating over all of them for each merge (up to 120 merges) is 5500*120 = 660k ops.
        # This is fine.
        
        pair_counts = Counter()
        for b_idx in bad_cycles:
            b_cyc_set = set(ALL_CYCLES[b_idx])
            for p in CYCLE_PAIRS[idx]:
                if p[0] in b_cyc_set and p[1] in b_cyc_set:
                    pair_counts[p] += 1
        
        # If no counts (shouldn't happen as current cycle is bad), fallback to random
        if not pair_counts:
             # Pick random pair from the 6
             pairs = CYCLE_PAIRS[idx]
             p = random.choice(pairs)
        else:
             # Pick pair with max count
             p = pair_counts.most_common(1)[0][0]
        
        _union(parent, p[0], p[1])
        
        # Update bad_cycles
        # Re-evaluate all cycles
        new_bad_cycles = []
        for b_idx in bad_cycles:
             if _is_bad_cycle(ALL_CYCLES[b_idx], parent):
                 new_bad_cycles.append(b_idx)
        bad_cycles = new_bad_cycles
        
    # Count colors
    colors = [_find(parent, i) for i in range(NUM_EDGES)]
    return len(set(colors))

def construct():
    start_time = time.time()
    best_count = 1
    best_parent = None
    
    # We need to return the colors, not just the count.
    # So we need to store the parent array of the best run.
    
    for seed in range(200):
        if time.time() - start_time > 55:
            break
            
        parent = list(range(NUM_EDGES))
        bad_cycles = list(range(NUM_CYCLES))
        
        while bad_cycles:
            idx = random.choice(bad_cycles)
            cyc = ALL_CYCLES[idx]
            
            pair_counts = Counter()
            # Optimization: Only count pairs from the current cycle
            # But we need to know how many *other* bad cycles they appear in.
            # So we still need to scan bad_cycles.
            
            # To speed up, we can precompute for each edge, the list of cycles it's in?
            # No, let's just do the scan.
            
            for b_idx in bad_cycles:
                b_cyc_set = set(ALL_CYCLES[b_idx])
                for p in CYCLE_PAIRS[idx]:
                    if p[0] in b_cyc_set and p[1] in b_cyc_set:
                        pair_counts[p] += 1
            
            if not pair_counts:
                pairs = CYCLE_PAIRS[idx]
                p = random.choice(pairs)
            else:
                p = pair_counts.most_common(1)[0][0]
            
            _union(parent, p[0], p[1])
            
            new_bad_cycles = []
            for b_idx in bad_cycles:
                if _is_bad_cycle(ALL_CYCLES[b_idx], parent):
                    new_bad_cycles.append(b_idx)
            bad_cycles = new_bad_cycles
            
        colors = [_find(parent, i) for i in range(NUM_EDGES)]
        count = len(set(colors))
        
        if count > best_count:
            best_count = count
            best_parent = parent[:]
            
    # Reconstruct colors from best_parent
    if best_parent is None:
         return [0] * NUM_EDGES
         
    final_colors = [_find(best_parent, i) for i in range(NUM_EDGES)]
    # Remap to 0..count-1
    unique_colors = list(set(final_colors))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    return [color_map[c] for c in final_colors]

# EVOLVE-BLOCK-END
