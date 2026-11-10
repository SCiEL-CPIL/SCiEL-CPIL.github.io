import time
import random
import itertools
import math

N = 16

# Precompute all triples
ALL_TRIPLES = []
TRIPLE_MASKS = []
TRIPLE_PAIRS = []

# Pair mapping: (a, b) -> index 0..119
# We can just use a set of tuples for used_pairs, it's simpler and fast enough for 120 pairs
# But let's precompute pair indices for each triple to make validity check O(1) with a boolean array or set

# Let's use a set of frozensets or just tuples for used pairs
# For speed, map pair (a,b) to an integer index
pair_to_idx = {}
idx_to_pair = []
idx = 0
for a in range(N):
    for b in range(a + 1, N):
        pair_to_idx[(a, b)] = idx
        idx_to_pair.append((a, b))
        idx += 1

NUM_PAIRS = idx

for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            mask = (1 << a) | (1 << b) | (1 << c)
            ALL_TRIPLES.append((a, b, c))
            TRIPLE_MASKS.append(mask)
            # Get pair indices
            p1 = pair_to_idx[(a, b)]
            p2 = pair_to_idx[(a, c)]
            p3 = pair_to_idx[(b, c)]
            TRIPLE_PAIRS.append((p1, p2, p3))

NUM_TRIPLES = len(ALL_TRIPLES)

def calc_alpha_exact(edge_indices):
    """
    Calculate exact independence number.
    """
    if not edge_indices:
        return N
    
    edge_masks = [TRIPLE_MASKS[i] for i in edge_indices]
    
    # Check from largest size down
    # Optimization: We can stop early if we find one
    for k in range(N, 0, -1):
        # Generate all subsets of size k
        for subset in itertools.combinations(range(N), k):
            mask_s = 0
            for v in subset:
                mask_s |= (1 << v)
            
            is_indep = True
            for em in edge_masks:
                if (mask_s & em) == em:
                    is_indep = False
                    break
            
            if is_indep:
                return k
    return 0

def has_independent_set_of_size(edge_indices, k):
    """
    Check if there exists an independent set of size k.
    Returns True if found, False otherwise.
    """
    if k == 0:
        return True
    if k > N:
        return False
        
    edge_masks = [TRIPLE_MASKS[i] for i in edge_indices]
    
    for subset in itertools.combinations(range(N), k):
        mask_s = 0
        for v in subset:
            mask_s |= (1 << v)
        
        is_indep = True
        for em in edge_masks:
            if (mask_s & em) == em:
                is_indep = False
                break
        
        if is_indep:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # State
    # used_pairs: set of pair indices
    # current_edges: list of triple indices
    
    best_alpha = N
    best_edges = []
    
    # Phase 1: Greedy Construction
    # We want a maximal linear system.
    # Random order of triples
    order = list(range(NUM_TRIPLES))
    random.shuffle(order)
    
    used_pairs = set()
    current_edges = []
    
    for t_idx in order:
        p1, p2, p3 = TRIPLE_PAIRS[t_idx]
        if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            current_edges.append(t_idx)
            
    # Calculate initial alpha
    current_alpha = calc_alpha_exact(current_edges)
    best_alpha = current_alpha
    best_edges = list(current_edges)
    
    # Phase 2: Local Search (Simulated Annealing style)
    # We try to minimize alpha.
    # Move: Remove one edge, add one compatible edge.
    
    # To speed up, we precompute a list of all triple indices
    all_triple_indices = list(range(NUM_TRIPLES))
    
    # Temperature for SA-like acceptance
    # We accept if alpha decreases, or with prob p if alpha stays same.
    # We reject if alpha increases (usually).
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # If we have few edges, we might want to add more?
        # The greedy phase should have made it maximal.
        # But local search swaps might keep it maximal (remove 1, add 1).
        
        if len(current_edges) == 0:
            break
            
        # 1. Pick a random edge to remove
        rem_idx_pos = random.randint(0, len(current_edges) - 1)
        rem_edge_idx = current_edges[rem_idx_pos]
        
        # Remove it
        # Update used_pairs
        rp1, rp2, rp3 = TRIPLE_PAIRS[rem_edge_idx]
        used_pairs.remove(rp1)
        used_pairs.remove(rp2)
        used_pairs.remove(rp3)
        current_edges.pop(rem_idx_pos)
        
        # 2. Find a compatible edge to add
        # A compatible edge is one whose 3 pairs are not in used_pairs
        # And it's not the one we just removed (though it's not in current_edges, 
        # its pairs are now free, so it IS compatible technically, but we want a *new* edge 
        # or maybe the same one? If we add the same one, nothing changes. 
        # So we should prefer different edges.
        
        candidates = []
        for t_idx in all_triple_indices:
            if t_idx == rem_edge_idx:
                continue
            p1, p2, p3 = TRIPLE_PAIRS[t_idx]
            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                candidates.append(t_idx)
                
        if not candidates:
            # No candidate found. Restore old edge.
            current_edges.append(rem_edge_idx)
            used_pairs.add(rp1)
            used_pairs.add(rp2)
            used_pairs.add(rp3)
            continue
            
        new_edge_idx = random.choice(candidates)
        
        # Add new edge
        np1, np2, np3 = TRIPLE_PAIRS[new_edge_idx]
        used_pairs.add(np1)
        used_pairs.add(np2)
        used_pairs.add(np3)
        current_edges.append(new_edge_idx)
        
        # 3. Evaluate new alpha
        # Optimization: Check if alpha decreased.
        # If has_independent_set_of_size(current_edges, current_alpha) is False, then new_alpha < current_alpha.
        
        if not has_independent_set_of_size(current_edges, current_alpha):
            # Alpha decreased
            # Find new alpha
            new_alpha = current_alpha - 1
            while new_alpha > 0 and has_independent_set_of_size(current_edges, new_alpha):
                new_alpha -= 1
            # Wait, the loop above finds the largest k such that independent set exists?
            # No, `has_independent_set_of_size` returns True if exists.
            # If it returns False for current_alpha, we know new_alpha < current_alpha.
            # We need to find the actual new_alpha.
            # It's likely current_alpha - 1.
            # Let's just calculate exact if it decreased, to be safe.
            # Or, we can just set new_alpha = calc_alpha_exact(current_edges)
            # Since this happens rarely, exact calc is fine.
            new_alpha = calc_alpha_exact(current_edges)
            
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_edges = list(current_edges)
                
            current_alpha = new_alpha
            # Accept move (it's an improvement)
            
        else:
            # Alpha is >= current_alpha
            # It's likely equal.
            # Accept with some probability to explore (Simulated Annealing)
            # Probability decreases over time?
            # Let's use a fixed probability for simplicity, e.g., 0.1
            if random.random() < 0.1:
                # Accept, alpha stays same (or increased, which is bad, but we assume equal)
                # We should verify it didn't increase.
                # If it increased, we should reject.
                # Checking for increase is expensive.
                # Let's assume 1-swap rarely increases alpha.
                # If we want to be safe:
                # new_alpha = calc_alpha_exact(current_edges)
                # if new_alpha > current_alpha: reject
                # if new_alpha == current_alpha: accept
                # if new_alpha < current_alpha: accept (and update best)
                
                # Let's do the exact check only if we accept the "equal" move?
                # No, let's just reject if we don't know.
                # Actually, if has_independent_set_of_size(current_alpha) is True, 
                # it's possible alpha increased.
                # But in linear hypergraphs on 16 points, alpha is small.
                # Let's just reject "equal" moves most of the time to focus on improvements,
                # but allow some exploration.
                
                # To be safe and fast, let's just reject if we didn't prove it decreased.
                # Wait, if we reject all equal moves, we get stuck.
                # Let's accept equal moves with prob 0.2.
                # We will NOT recalculate alpha exactly for accepted equal moves, 
                # we assume it's current_alpha.
                # This might drift if alpha actually increased, but that's unlikely 
                # and we have a "best" tracker.
                
                current_alpha = current_alpha # Unchanged
            else:
                # Reject: Restore old edge
                current_edges.pop()
                used_pairs.remove(np1)
                used_pairs.remove(np2)
                used_pairs.remove(np3)
                
                current_edges.append(rem_edge_idx)
                used_pairs.add(rp1)
                used_pairs.add(rp2)
                used_pairs.add(rp3)
                
    # Return the best system found
    return [list(ALL_TRIPLES[i]) for i in best_edges]

# Quick sanity check logic for has_independent_set_of_size
# If current_alpha is 5, and we check size 5.
# If no set of size 5 exists, alpha is 4 or less.
# Then we calc_exact.

# One optimization: 
# In the greedy phase, we stop when no more edges can be added.
# This ensures maximality.

# The local search maintains maximality (remove 1, add 1).

# The main bottleneck is `has_independent_set_of_size` and `calc_alpha_exact`.
# `has_independent_set_of_size` for k=5: C(16,5) = 4368 subsets.
# Each check is ~30 edges. 4368 * 30 = 131k ops.
# This is very fast in Python (~0.01s).
# So we can do ~1000-2000 iterations in 120s.
# This is a significant improvement over the previous program which did fewer iterations 
# due to slower compatibility checks and full alpha recalcs.
