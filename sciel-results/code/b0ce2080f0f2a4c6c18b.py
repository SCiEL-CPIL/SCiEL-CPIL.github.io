import time
import numpy as np
from scipy.optimize import differential_evolution
import sys

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if len(arr) == 0:
        return 0.0
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    # Use a threshold to handle floating point errors
    sumset_size = np.sum(S_sum[:N] > 1e-6)
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def _local_search(U, max_iters=2000, time_limit=5.0):
    """
    Refine a discrete set U using local search.
    Moves: add an element, remove an element, or swap an element.
    """
    U_set = set(U)
    if 0 not in U_set:
        U_set.add(0)
    
    # Determine search range
    if not U_set:
        return [0]
    max_val = max(U_set)
    # Allow expansion slightly beyond current max
    search_max = max_val + 10 
    if search_max < 10:
        search_max = 10
    
    current_score = _score_set(list(U_set))
    best_score = current_score
    best_set = U_set.copy()
    
    start_time = time.time()
    np.random.seed(42)
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a random move
        move_type = np.random.random()
        
        if move_type < 0.4:
            # Add
            if len(U_set) < 200: # Cap size for speed
                x = np.random.randint(0, search_max + 1)
                if x not in U_set:
                    new_set = U_set | {x}
                    if 0 not in new_set: continue
                    s = _score_set(list(new_set))
                    if s > best_score:
                        best_score = s
                        best_set = new_set
                        U_set = new_set
                        # Expand search range if we added a large number
                        if x > max_val:
                            max_val = x
                            search_max = max_val + 10
        elif move_type < 0.7:
            # Remove
            if len(U_set) > 2: # Keep at least 0 and one other
                # Don't remove 0
                removable = [x for x in U_set if x != 0]
                if removable:
                    x = np.random.choice(removable)
                    new_set = U_set - {x}
                    s = _score_set(list(new_set))
                    if s > best_score:
                        best_score = s
                        best_set = new_set
                        U_set = new_set
        else:
            # Swap
            if len(U_set) > 1:
                removable = [x for x in U_set if x != 0]
                if removable:
                    x_out = np.random.choice(removable)
                    x_in = np.random.randint(0, search_max + 1)
                    if x_in not in U_set:
                        new_set = (U_set - {x_out}) | {x_in}
                        if 0 in new_set:
                            s = _score_set(list(new_set))
                            if s > best_score:
                                best_score = s
                                best_set = new_set
                                U_set = new_set
                                if x_in > max_val:
                                    max_val = x_in
                                    search_max = max_val + 10
                                
    return list(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Strategy:
    # 1. Use Differential Evolution on a continuous relaxation.
    #    We define a grid of size M. The variable x is a vector of length M.
    #    We interpret x_i as a "weight" for element i.
    #    Objective: Maximize a surrogate for log(|U-U|/|U+U|).
    #    Since we can't easily compute |U+U| from continuous weights without
    #    thresholding, we use a proxy:
    #    Let p be the probability vector (sigmoid(x)).
    #    Approximate support size of convolution using L1 norm of expected convolution?
    #    Actually, a simpler structural approach is to use DE to find the "best"
    #    discrete set directly by encoding the set as a bitstring (0/1) but
    #    relaxing the bounds to [0,1] and using a penalty for non-binary values?
    #    DE works on continuous vectors.
    
    # Alternative Structural Change:
    # The current program uses "product sets". Let's try "additive combinatorics"
    # heuristics directly.
    # Known good sets are often "Sidon-like" or "Golomb Ruler" variants,
    # or sets that are unions of arithmetic progressions with specific gaps.
    
    # However, the prompt asks for a DIFFERENT strategy.
    # Let's try a **Genetic Algorithm** with a different representation:
    # Represent U as a binary vector of length N_max.
    # Fitness = Score.
    # Crossover: Bitwise OR/AND/XOR with masking.
    # Mutation: Flip bits.
    # This is a standard bitstring GA, distinct from the mixed-radix search.
    
    N_max = 200 # Search space size
    pop_size = 50
    max_gens = 50
    
    # Initialize population with diverse random sets
    population = []
    for i in range(pop_size):
        # Random size between 5 and 30
        size = np.random.randint(5, 30)
        # Random elements from 0 to N_max
        elems = np.random.choice(N_max, size=size, replace=False)
        if 0 not in elems:
            elems[0] = 0
        elems = np.unique(elems)
        population.append(elems)
        
    best_set = population[0]
    best_score = _score_set(best_set)
    
    # Add some heuristic seeds
    # Golomb ruler approximation
    def golomb_like(n):
        # Greedy Golomb ruler
        marks = [0]
        for i in range(1, n):
            # Find next mark that doesn't create duplicate differences
            candidate = 1
            found = False
            while candidate < N_max:
                ok = True
                for m in marks:
                    diff = candidate - m
                    # Check if this difference exists
                    for m2 in marks:
                        if m2 != m and abs(m - m2) == diff:
                            ok = False
                            break
                        if m2 != m and abs(m2 - m) == diff: # redundant but safe
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    marks.append(candidate)
                    found = True
                    break
                candidate += 1
            if not found:
                break
        return marks
    
    # Add Golomb rulers of various sizes
    for size in [5, 10, 15, 20]:
        g = golomb_like(size)
        if len(g) >= 2:
            population.append(g)
            
    # Add "Sparse AP" seeds: {0, 1, L, L+1, 2L, ...}
    for L in [3, 5, 7, 10]:
        elems = []
        for k in range(5):
            elems.append(k*L)
            elems.append(k*L + 1)
        elems = [x for x in elems if x < N_max]
        if 0 in elems:
            population.append(np.unique(elems))
            
    # Evaluate initial population
    for i, ind in enumerate(population):
        s = _score_set(ind)
        if s > best_score:
            best_score = s
            best_set = ind
            
    # Genetic Algorithm Loop
    np.random.seed(123)
    
    for gen in range(max_gens):
        if time.time() > deadline - 20:
            break
            
        # Evaluate all
        scores = [_score_set(ind) for ind in population]
        total_score = sum(scores)
        
        # Update best
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_set = population[max_idx]
            
        # Create new population
        new_population = []
        
        # Elitism: keep best
        new_population.append(population[max_idx])
        
        while len(new_population) < pop_size:
            # Selection: Tournament
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if scores[idx1] > scores[idx2]:
                p1 = population[idx1]
            else:
                p1 = population[idx2]
                
            idx3 = np.random.randint(0, pop_size)
            idx4 = np.random.randint(0, pop_size)
            if scores[idx3] > scores[idx4]:
                p2 = population[idx3]
            else:
                p2 = population[idx4]
                
            # Crossover: Bitwise on fixed length vector
            # Convert to binary vector of length N_max
            v1 = np.zeros(N_max, dtype=int)
            v1[p1] = 1
            v2 = np.zeros(N_max, dtype=int)
            v2[p2] = 1
            
            # Uniform crossover
            mask = np.random.random(N_max) > 0.5
            child = np.where(mask, v1, v2)
            
            # Ensure 0 is in the set
            child[0] = 1
            
            # Mutation
            # Flip some bits
            num_mut = np.random.randint(1, 5)
            for _ in range(num_mut):
                idx = np.random.randint(0, N_max)
                if idx == 0: continue
                child[idx] = 1 - child[idx]
                
            # Convert back to list
            child_list = np.where(child > 0)[0]
            
            # Optional: Local search on the best child of this generation?
            # To save time, only do local search on the global best occasionally
            # or on the child if it looks promising.
            
            new_population.append(child_list)
            
        population = new_population
        
    # Final Local Search on the best found set
    if time.time() < deadline - 5:
        best_set = _local_search(best_set, max_iters=2000, time_limit=10.0)
        
    return list(best_set)

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
