"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    n = 12
    m = 40
    total_subsets = 1 << n
    
    # 1. Generate all subsets as lists of integers, and also keep as frozensets for logic
    # We need to map indices to subsets.
    subsets = []
    for mask in range(total_subsets):
        s = [i for i in range(n) if mask & (1 << i)]
        subsets.append(s)
        
    # Precompute comparability matrix
    # M[i, j] is 1 if subsets[i] and subsets[j] are comparable (one is strict subset of other)
    # Note: M[i, i] = 0. M is symmetric.
    M = [[0] * total_subsets for _ in range(total_subsets)]
    
    # Optimization: Only compute for i < j? Or use bitmasks?
    # 4096^2 = 16M operations. In Python, this might be slow if done naively in nested loops with set operations.
    # Let's optimize the precomputation.
    # Convert to frozensets for fast issubset?
    fsubsets = [frozenset(s) for s in subsets]
    
    for i in range(total_subsets):
        for j in range(i + 1, total_subsets):
            if len(fsubsets[i]) < len(fsubsets[j]):
                if fsubsets[i].issubset(fsubsets[j]):
                    M[i][j] = 1
                    M[j][i] = 1
            elif len(fsubsets[i]) > len(fsubsets[j]):
                if fsubsets[j].issubset(fsubsets[i]):
                    M[i][j] = 1
                    M[j][i] = 1
            # If same length, they cannot be strict subsets of each other unless equal, but indices are distinct.
            
    # 2. Initialize with Initial Segment (smallest sets)
    # Sort indices by size of subset, then by index
    indices_by_size = sorted(range(total_subsets), key=lambda x: (len(subsets[x]), x))
    current_family = indices_by_size[:m]
    current_set_indices = set(current_family)
    
    # Calculate initial score
    def get_score(family_indices):
        score = 0
        for i in range(len(family_indices)):
            for j in range(i + 1, len(family_indices)):
                score += M[family_indices[i]][family_indices[j]]
        return score

    current_score = get_score(current_family)
    
    # 3. Simulated Annealing
    T = 10.0
    cooling = 0.995
    iterations = 20000
    rng = random.Random(42) # Fixed seed for reproducibility in eval? Task says "seed program" but usually we want robustness. 
    # The task says "search score is value/seed". It implies the seed is a random seed or the seed program's score? 
    # "1.0 equals the seed program". 
    # I will use a fixed seed to ensure deterministic behavior for the "child" to be comparable, 
    # or just rely on the structure. Actually, random is fine.
    
    for _ in range(iterations):
        if T < 0.01:
            break
            
        # Pick a random member to remove
        i = rng.randrange(m)
        remove_idx = current_family[i]
        
        # Pick a random candidate to add (not in current family)
        # To avoid infinite loops or slow selection, we can pick a random index from 0..4095 and check if it's in the set.
        # Probability of collision is 40/4096 ~ 1%. So expected tries is small.
        add_idx = rng.randrange(total_subsets)
        while add_idx in current_set_indices:
            add_idx = rng.randrange(total_subsets)
            
        # Calculate Delta
        # Loss from removing remove_idx: sum of M[remove_idx][other] for other in family (except remove_idx)
        loss = 0
        for j in range(m):
            if current_family[j] != remove_idx:
                loss += M[remove_idx][current_family[j]]
                
        # Gain from adding add_idx: sum of M[add_idx][other] for other in family (except remove_idx, since it's gone)
        # Wait, the "other" set is the family MINUS remove_idx.
        gain = 0
        for j in range(m):
            if current_family[j] != remove_idx:
                gain += M[add_idx][current_family[j]]
                
        delta = gain - loss
        
        if delta > 0:
            # Accept
            current_family[i] = add_idx
            current_set_indices.remove(remove_idx)
            current_set_indices.add(add_idx)
            current_score += delta
        else:
            # Accept with probability exp(delta / T)
            if rng.random() < math.exp(delta / T):
                current_family[i] = add_idx
                current_set_indices.remove(remove_idx)
                current_set_indices.add(add_idx)
                current_score += delta
                
        T *= cooling

    # Return the result as lists of integers
    result = [subsets[idx] for idx in current_family]
    
    # Ensure types are correct
    # subsets are already lists of ints
    return result

# EVOLVE-BLOCK-END
