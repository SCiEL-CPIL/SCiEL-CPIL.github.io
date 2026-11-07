"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    """
    Count the number of unordered pairs (i, j) with i < j such that 
    sets[i] is a strict subset of sets[j] or sets[j] is a strict subset of sets[i].
    Uses bitmask representation for efficiency.
    """
    n = P['n']
    m = len(sets)
    masks = [0] * m
    for i in range(m):
        mask = 0
        for v in sets[i]:
            if 0 <= v < n:
                mask |= (1 << v)
        masks[i] = mask
    
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            inter = mi & mj
            if inter == mi and mi != mj:
                count += 1
            elif inter == mj and mi != mj:
                count += 1
    return count

def construct():
    n = P['n']
    m = P['m']
    
    start_time = time.time()
    max_time = 85.0
    
    # Strategy: 
    # 1. Create a "backbone" chain of 13 sets: C_0, C_1, ..., C_12
    #    where C_k is the set of the first k elements of a random permutation.
    #    This chain contributes 12*13/2 = 78 comparable pairs internally.
    # 2. We need 27 more sets. 
    #    Ideally, these sets should be comparable with as many backbone sets as possible.
    #    A set S is comparable with C_k if S is a subset of C_k or C_k is a subset of S.
    #    Note: If S is not a subset of the universe's permutation structure in a "nested" way, 
    #    it might not be comparable with many C_k.
    #    
    #    Better Idea: 
    #    Instead of a single chain, consider that the maximum number of comparable pairs 
    #    in a family of size m from a universe of size n is achieved by taking a chain 
    #    as long as possible and then adding sets that are comparable to many members of the chain.
    #    
    #    Let's try a "Layered" approach:
    #    Generate a random total order of the 12 elements: e0, e1, ..., e11.
    #    Define P_k = {e0, ..., e_{k-1}} for k=0..12. These form a chain.
    #    
    #    Now, for the remaining 27 sets, we want to pick sets S such that S is comparable 
    #    with many P_k. 
    #    S is comparable with P_k if S subset P_k or P_k subset S.
    #    If we pick S to be of the form P_a union {some elements not in P_a} or P_a minus {some elements}, 
    #    we might maintain comparability.
    #    
    #    Actually, a stronger structure is a "lattice" or "grid". 
    #    Consider splitting the 12 elements into two groups A and B of size 6.
    #    The subsets of A form a chain of length 7. The subsets of B form a chain of length 7.
    #    The product of these chains is a grid of 49 sets.
    #    Any two sets in this grid are comparable if and only if they are comparable in both dimensions? 
    #    No, (A1, B1) and (A2, B2) are comparable if A1 subset A2 AND B1 subset B2, OR A2 subset A1 AND B2 subset B1.
    #    This is a poset of dimension 2.
    #    We can pick 40 sets from this 7x7 grid.
    #    
    #    Let's evaluate the potential of the 7x7 grid (36 elements: 6 in A, 6 in B).
    #    Total sets in grid: 49.
    #    We need 40 distinct sets.
    #    Let's see how many comparable pairs are in the full 7x7 grid.
    #    In a chain of length L, pairs = L(L-1)/2.
    #    In a product of two chains C_a x C_b, the number of comparable pairs is:
    #    Sum over all pairs (x,y) where x <= y.
    #    For a grid of size H x W (chains of length H and W), the number of comparable pairs is:
    #    Sum_{i=1..H} Sum_{j=1..W} (i-1)*(j-1) + ... ? No.
    #    Let's count pairs (u, v) with u < v in the poset.
    #    u = (a1, b1), v = (a2, b2). u < v iff a1 < a2 and b1 < b2, OR a1 > a2 and b1 > b2? No, that's not right.
    #    u <= v iff a1 <= a2 AND b1 <= b2.
    #    So we count pairs where (a1 < a2 and b1 < b2) OR (a1 > a2 and b1 > b2) OR (a1 == a2 and b1 < b2) OR (a1 < a2 and b1 == b2).
    #    Actually, just count all pairs (u,v) with u != v and u <= v.
    #    This is equivalent to counting all pairs of indices (i,j) in the grid such that i1 <= i2 and j1 <= j2, excluding i1==i2 and j1==j2.
    #    Let's compute this for 7x7.
    #    Total pairs = 49*48/2 = 1176.
    #    Comparable pairs = ?
    #    Let's write a quick mental script or estimate.
    #    For a 2x2 grid: (0,0)<(0,1), (0,0)<(1,0), (0,0)<(1,1), (0,1)<(1,1), (1,0)<(1,1). Total 5.
    #    For 7x7, it should be significantly higher than a random set of 40.
    #    
    #    Let's implement the 7x7 grid strategy.
    #    1. Split 12 elements into A (6) and B (6).
    #    2. Generate all 49 combinations of subsets of A and subsets of B.
    #    3. Calculate the score of the full 49 sets? No, we only need 40.
    #    4. Which 40 to pick? 
    #       We want to maximize comparable pairs among the chosen 40.
    #       This is a Maximum Comparable Subset problem on a grid poset.
    #       Greedy approach: Start with the 49 sets, compute the "degree" (number of comparable neighbors) for each.
    #       Iteratively remove the set with the lowest degree until 40 remain.
    #       Or, start with empty, add the set that adds the most comparable pairs with existing ones.
    #    
    #    Let's try the "Remove Lowest Degree" heuristic on the full 49-set grid.
    
    # Step 1: Split elements
    elements = list(range(n))
    random.shuffle(elements)
    A = elements[:6]
    B = elements[6:]
    
    # Step 2: Generate all subsets of A and B
    subs_A = []
    for mask in range(1 << 6):
        s = []
        for i in range(6):
            if mask & (1 << i):
                s.append(A[i])
        subs_A.append(tuple(s))
        
    subs_B = []
    for mask in range(1 << 6):
        s = []
        for i in range(6):
            if mask & (1 << i):
                s.append(B[i])
        subs_B.append(tuple(s))
        
    # Step 3: Construct the 49 sets
    # Set (i, j) is subs_A[i] union subs_B[j]
    all_sets = []
    for sa in subs_A:
        for sb in subs_B:
            s = list(sa) + list(sb)
            all_sets.append(s)
            
    # Step 4: We have 49 sets. We need 40.
    # Let's compute the comparability graph of these 49 sets.
    # Node i is comparable to node j if one is a strict subset of the other.
    # We want to select a subset of 40 nodes that maximizes the number of edges.
    # Heuristic: 
    # 1. Compute degree of each node in the full graph.
    # 2. Remove the node with the minimum degree.
    # 3. Recalculate degrees? Or just remove 9 nodes with lowest initial degrees?
    #    Removing a node reduces the degree of its neighbors.
    #    Let's do iterative removal.
    
    current_set_indices = list(range(49))
    # Precompute masks for the 49 sets
    masks = [0] * 49
    for i, s in enumerate(all_sets):
        msk = 0
        for v in s:
            msk |= (1 << v)
        masks[i] = msk
        
    def is_comparable(m1, m2):
        if m1 == m2:
            return False
        return (m1 & m2) == m1 or (m1 & m2) == m2
        
    # Build adjacency list
    adj = [[] for _ in range(49)]
    for i in range(49):
        for j in range(i + 1, 49):
            if is_comparable(masks[i], masks[j]):
                adj[i].append(j)
                adj[j].append(i)
                
    # Iteratively remove the node with the lowest degree among the remaining nodes
    # until we have 40 nodes.
    # Note: The degree in the induced subgraph is what matters.
    
    remaining = set(range(49))
    
    # To speed up, we can just simulate the removal.
    # 9 removals.
    for _ in range(9):
        min_deg = 1000
        min_node = -1
        for node in remaining:
            deg = 0
            for neighbor in adj[node]:
                if neighbor in remaining:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                min_node = node
        remaining.remove(min_node)
        
    final_indices = list(remaining)
    final_sets = [all_sets[i] for i in final_indices]
    
    # Verify distinctness (should be distinct by construction)
    # Verify count
    if len(final_sets) != m:
        # Fallback if something went wrong
        final_sets = all_sets[:m]
        
    # Convert to lists of ints
    result = []
    for s in final_sets:
        result.append([int(x) for x in s])
        
    # Calculate score for debugging/verification
    score = _count_comparable_pairs(result)
    # print(f"Grid Strategy Score: {score}")
    
    # Is this better than random? 
    # Let's also try a simple local search on this result to see if we can improve.
    # But the prompt asks for a DIFFERENT strategy. The grid construction IS the strategy.
    # However, a small local search on the grid result might be safe and improve it.
    # Let's add a very short local search (SA) on the grid result to polish it.
    
    start_time_search = time.time()
    T = 10.0
    cooling = 0.99
    min_T = 0.1
    current_sets = final_sets
    current_score = score
    best_sets = [list(s) for s in current_sets]
    best_score = current_score
    
    # Precompute masks for current sets
    masks = [0] * m
    for i in range(m):
        msk = 0
        for v in current_sets[i]:
            msk |= (1 << v)
        masks[i] = msk
        
    def update_masks(idx, new_s):
        msk = 0
        for v in new_s:
            msk |= (1 << v)
        masks[idx] = msk
        
    def calc_score_fast():
        count = 0
        for i in range(m):
            mi = masks[i]
            for j in range(i + 1, m):
                mj = masks[j]
                inter = mi & mj
                if inter == mi and mi != mj:
                    count += 1
                elif inter == mj and mi != mj:
                    count += 1
        return count
        
    # Local Search
    while T > min_T and time.time() - start_time < max_time:
        idx = random.randint(0, m - 1)
        old_s = current_sets[idx]
        old_mask = masks[idx]
        
        # Generate a new set
        # Try to swap an element or change size
        # Simple move: pick a random element in the set and remove it, or pick a random element outside and add it.
        r = random.random()
        if r < 0.5:
            # Remove an element
            if len(old_s) > 0:
                elem_to_remove = random.choice(old_s)
                new_s = [x for x in old_s if x != elem_to_remove]
            else:
                continue
        else:
            # Add an element
            missing = [v for v in range(n) if v not in set(old_s)]
            if len(missing) > 0:
                elem_to_add = random.choice(missing)
                new_s = old_s + [elem_to_add]
            else:
                continue
                
        new_s_tuple = tuple(sorted(new_s))
        old_s_tuple = tuple(sorted(old_s))
        
        if new_s_tuple == old_s_tuple:
            continue
            
        # Check distinctness
        is_dup = False
        for k in range(m):
            if k != idx and tuple(sorted(current_sets[k])) == new_s_tuple:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Apply
        current_sets[idx] = new_s
        update_masks(idx, new_s)
        new_score = calc_score_fast()
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_sets = [list(s) for s in current_sets]
        else:
            # Revert
            current_sets[idx] = old_s
            masks[idx] = old_mask
            
        T *= cooling
        
    result = []
    for s in best_sets:
        result.append([int(x) for x in s])
        
    return result
# EVOLVE-BLOCK-END
