import random
from itertools import combinations

def construct():
    # 1. Generate all 4-subsets of [1, 16]
    all_sets = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    
    # 2. Precompute pairwise intersections
    # inter[i][j] is the bitwise AND of all_sets[i] and all_sets[j]
    # Using a list of lists of ints
    inter = [[0] * n_sets for _ in range(n_sets)]
    for i in range(n_sets):
        mi = all_sets[i]
        for j in range(i, n_sets):
            mj = all_sets[j]
            val = mi & mj
            inter[i][j] = val
            inter[j][i] = val

    def is_valid_add(c_idx, family):
        # family is a list of indices
        c_mask = all_sets[c_idx]
        # Check against all pairs in family
        # We can optimize by checking if c_idx is already in family (shouldn't be called if it is)
        for i in range(len(family)):
            ai = family[i]
            aci = inter[ai][c_idx]
            for j in range(i + 1, len(family)):
                aj = family[j]
                if inter[ai][aj] == aci == inter[aj][c_idx]:
                    return False
        return True

    def greedy_build(order):
        family = []
        family_set = set()
        for idx in order:
            if is_valid_add(idx, family):
                family.append(idx)
                family_set.add(idx)
        return family

    def local_search(family):
        family_set = set(family)
        improved = True
        while improved:
            improved = False
            # Try to add
            # We can iterate through all sets not in family
            # To speed up, we can just iterate all and check membership
            for c_idx in range(n_sets):
                if c_idx not in family_set:
                    if is_valid_add(c_idx, family):
                        family.append(c_idx)
                        family_set.add(c_idx)
                        improved = True
                        # Restart scanning? Or continue? 
                        # Adding one might block others. 
                        # Breaking and restarting is safer for finding all adds.
                        break
            
            if improved:
                continue
            
            # Try to swap
            # For each element in family, try to swap with a random element from complement
            # To be thorough, we might want to try all swaps, but that's O(N^2).
            # Let's try a few random swaps per element.
            swaps_done = 0
            for i in range(len(family)):
                a_idx = family[i]
                # Try up to 10 random candidates for swap
                for _ in range(10):
                    b_idx = random.randint(0, n_sets - 1)
                    if b_idx in family_set:
                        continue
                    # Check if swapping a_idx with b_idx is valid
                    # New family: family without a_idx, plus b_idx
                    # We need to check if b_idx is valid with respect to family \ {a_idx}
                    # AND that the rest of the family doesn't form a sunflower with b_idx
                    # is_valid_add(b_idx, family_without_a)
                    
                    # Optimization: 
                    # The only new triples involving b_idx are (b_idx, x, y) where x, y in family \ {a_idx}.
                    # The triples involving a_idx are removed.
                    # So we just need to check if b_idx is valid with respect to family \ {a_idx}.
                    
                    # Constructing the list family_without_a is O(N).
                    # Checking is O(N^2).
                    # Total O(N^3) per swap check? No, is_valid_add is O(M^2) where M is size of family.
                    # So O(N^2) per check.
                    # 10 checks * N elements * O(N^2) = 10 * N^3.
                    # If N=200, 8,000,000 ops per local search step?
                    # This might be too slow.
                    
                    # Let's optimize is_valid_add for the swap case.
                    # We can just call is_valid_add(b_idx, [x for x in family if x != a_idx])
                    # But creating the list is slow.
                    
                    # Alternative: 
                    # Check if b_idx forms a sunflower with any pair in family \ {a_idx}.
                    # This is what is_valid_add does.
                    
                    # Let's just do it, but limit the number of swap attempts.
                    
                    # To avoid creating a new list, we can modify is_valid_add to accept an "exclude" index.
                    pass
                    
            # This inner loop is getting complex and slow.
            # Let's simplify: 
            # 1. Add phase: Greedily add all possible.
            # 2. Swap phase: Try random swaps.
            
            # Let's implement a helper for swap check that is slightly more efficient
            # by not creating the sub-list if possible? 
            # Actually, just creating the list is fine for N=200.
            
            # Let's try a few random swaps in total, not per element.
            for _ in range(50):
                if not family:
                    break
                i = random.randint(0, len(family) - 1)
                a_idx = family[i]
                b_idx = random.randint(0, n_sets - 1)
                if b_idx in family_set:
                    continue
                
                # Create new family list without a_idx
                new_family = [x for j, x in enumerate(family) if j != i]
                
                if is_valid_add(b_idx, new_family):
                    family[i] = b_idx
                    family_set.remove(a_idx)
                    family_set.add(b_idx)
                    improved = True
                    break # Restart local search loop after a successful swap
            
        return family

    best_family = []
    best_size = 0
    
    # Iterated Local Search
    for restart in range(50):
        random.seed(restart)
        order = list(range(n_sets))
        random.shuffle(order)
        
        family = greedy_build(order)
        family = local_search(family)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family.copy()
            
        # Perturb best solution for next restart? 
        # Or just restart from scratch?
        # Let's try restarting from scratch with different seeds.
        # 50 restarts might be enough.
        
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
