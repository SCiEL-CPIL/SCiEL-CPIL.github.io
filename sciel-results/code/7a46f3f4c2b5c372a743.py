# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

M = 205

def _compute_forbidden():
    forbidden = set()
    for x in range(1, M):
        forbidden.add(pow(x, 2, M))
    forbidden.discard(0)
    return forbidden

FORBIDDEN = _compute_forbidden()

# Precompute conflict bitmasks
# conflict_mask[i] is a bitmask where bit j is 1 if i and j conflict
CONFLICT_MASKS = [0] * M
for a in range(M):
    mask = 0
    for b in range(M):
        if a != b and (a - b) % M in FORBIDDEN:
            mask |= (1 << b)
    CONFLICT_MASKS[a] = mask

def _count_bits(mask):
    return bin(mask).count('1')

def _max_independent_set_bnb(start_mask, deadline):
    """
    Branch and Bound to find the Maximum Independent Set.
    Returns the best mask found.
    """
    best_mask = start_mask
    best_size = _count_bits(start_mask)
    
    # Global variables for the recursive search
    # We use a list to allow modification in nested scope if needed, but we'll use nonlocal or just return best
    # Actually, standard recursion with return values is cleaner but slower due to overhead.
    # Let's use a class or mutable container for best_size and best_mask to avoid passing around.
    
    class Solver:
        def __init__(self, deadline):
            self.deadline = deadline
            self.best_size = best_size
            self.best_mask = best_mask
            self.calls = 0
            
        def solve(self, C, S_size, S_mask):
            if time.time() > self.deadline:
                return
            
            self.calls += 1
            
            # Bound: Even if we add all remaining candidates, can we beat best?
            # Simple bound: S_size + popcount(C)
            if S_size + _count_bits(C) <= self.best_size:
                return
            
            # Heuristic Bound: Greedy MIS on C
            # This provides a tighter upper bound than just |C|
            # We approximate the size of MIS(C)
            greedy_C = C
            greedy_size = 0
            while greedy_C:
                # Pick vertex with min degree in greedy_C? Or just any?
                # For upper bound, we want to pack as many as possible greedily?
                # Actually, a greedy *independent* set is a lower bound on the size of MIS.
                # Wait, we need an UPPER bound on the size of MIS(C) to prune.
                # Greedy coloring of the *complement* graph (which is the conflict graph restricted to C)
                # gives an upper bound? No, chromatic number of conflict graph is lower bound on clique number?
                # We are looking for Independent Set in G. This is Clique in G_complement.
                # Or just use the fact that MIS(G) <= |V| / (avg_degree + 1) is weak.
                # Let's stick to the simple bound first. If it's too slow, we add a better one.
                # For N=205, the simple bound might be sufficient with good vertex ordering.
                
                # Let's try a simple greedy independent set on C to see if it improves our current best?
                # No, that's for finding a solution, not bounding.
                
                # Let's just proceed with branching.
                break # Placeholder, we don't do greedy bound here for simplicity in first pass
                
            if C == 0:
                if S_size > self.best_size:
                    self.best_size = S_size
                    self.best_mask = S_mask
                return
            
            # Select vertex v from C
            # Heuristic: Max degree in G[C]
            best_v = -1
            max_deg = -1
            temp_C = C
            while temp_C:
                lsb = temp_C & -temp_C
                v = lsb.bit_length() - 1
                # Degree of v in G[C] is popcount(CONFLICT_MASKS[v] & C)
                deg = _count_bits(CONFLICT_MASKS[v] & C)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp_C ^= lsb
            
            v = best_v
            if v == -1:
                return
                
            v_bit = 1 << v
            neighbors = CONFLICT_MASKS[v]
            
            # Branch 1: Include v
            # New S: S_mask | v_bit
            # New C: C without v and without neighbors of v
            new_C1 = C & ~(neighbors | v_bit)
            self.solve(new_C1, S_size + 1, S_mask | v_bit)
            
            # Branch 2: Exclude v
            # New S: S_mask
            # New C: C without v
            new_C2 = C & ~v_bit
            self.solve(new_C2, S_size, S_mask)
            
    solver = Solver(deadline)
    solver.solve(start_mask, _count_bits(start_mask), start_mask)
    
    return solver.best_mask

def _greedy_init():
    """
    Fast greedy initialization using random order.
    """
    elements = list(range(M))
    random.shuffle(elements)
    mask = 0
    conflict_union = 0
    
    for e in elements:
        # Check if e is compatible with current mask
        if not (conflict_union & (1 << e)):
            mask |= (1 << e)
            conflict_union |= CONFLICT_MASKS[e]
    return mask

def furstenberg_sarkozy_205() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    random.seed(42)
    
    # Generate a few good initial guesses using greedy
    best_init_mask = 0
    best_init_size = 0
    for i in range(20):
        mask = _greedy_init()
        size = _count_bits(mask)
        if size > best_init_size:
            best_init_size = size
            best_init_mask = mask
            
    # Run B&B starting from the best greedy solution
    # The B&B will explore the space, potentially finding a larger set.
    # Since we start with a valid independent set, best_size is at least best_init_size.
    # The search space is defined by the candidates NOT in the initial set?
    # No, standard B&B starts with C = All Vertices, S = Empty.
    # However, if we start with C = All Vertices, it might take longer.
    # We can fix the initial set? No, that restricts the search.
    # Better: Start with C = All Vertices, S = Empty.
    # But we can use the greedy solution to set the initial best_size/best_mask.
    
    all_mask = (1 << M) - 1
    
    # To speed up, we can also try to run B&B from the greedy solution's complement?
    # No, just run standard B&B.
    # Optimization: If the graph is small, B&B is very fast.
    
    res_mask = _max_independent_set_bnb(all_mask, deadline)
    res_size = _count_bits(res_mask)
    
    # Fallback if B&B didn't find anything better (or if it timed out immediately?)
    # It should always find at least the empty set or the initial best if we pass it.
    # Wait, my B&B implementation above initializes best_size with start_mask.
    # If I pass all_mask, best_size starts at 0.
    # I should pass the greedy mask as the initial best, but C should be all_mask.
    
    # Let's modify _max_independent_set_bnb to accept an initial best mask/size separately from C.
    # Or just call it with start_mask = all_mask, but initialize the solver with the greedy result.
    
    # Let's re-run with a wrapper that allows seeding the best solution.
    
    class Solver:
        def __init__(self, deadline, init_best_mask, init_best_size):
            self.deadline = deadline
            self.best_size = init_best_size
            self.best_mask = init_best_mask
            
        def solve(self, C, S_size, S_mask):
            if time.time() > self.deadline:
                return
            
            # Bound
            if S_size + _count_bits(C) <= self.best_size:
                return
            
            if C == 0:
                if S_size > self.best_size:
                    self.best_size = S_size
                    self.best_mask = S_mask
                return
            
            # Select vertex v from C with max degree in G[C]
            best_v = -1
            max_deg = -1
            temp_C = C
            while temp_C:
                lsb = temp_C & -temp_C
                v = lsb.bit_length() - 1
                deg = _count_bits(CONFLICT_MASKS[v] & C)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp_C ^= lsb
            
            v = best_v
            if v == -1:
                return
                
            v_bit = 1 << v
            neighbors = CONFLICT_MASKS[v]
            
            # Branch 1: Include v
            new_C1 = C & ~(neighbors | v_bit)
            self.solve(new_C1, S_size + 1, S_mask | v_bit)
            
            # Branch 2: Exclude v
            new_C2 = C & ~v_bit
            self.solve(new_C2, S_size, S_mask)
            
    solver = Solver(deadline, best_init_mask, best_init_size)
    solver.solve(all_mask, 0, 0)
    
    final_mask = solver.best_mask
    
    # Convert mask to array
    result = []
    m = final_mask
    i = 0
    while m:
        if m & 1:
            result.append(i)
        m >>= 1
        i += 1
        
    if not result:
        return np.array([0], dtype=np.int64)
        
    return np.array(sorted(result), dtype=np.int64)

# EVOLVE-BLOCK-END
