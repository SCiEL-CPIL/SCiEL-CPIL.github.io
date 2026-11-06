"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute LCMs
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # We need to avoid three distinct elements a, b, c such that:
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # Let L be this value.
    # This implies a, b, c are divisors of L.
    # Also, lcm(a,b)=L means that for every prime power p^k || L, 
    # max(v_p(a), v_p(b)) = k.
    # If we fix L, the set of numbers in [1, N] that divide L is fixed.
    # Within that set, we want to avoid triples.
    
    # Actually, a more direct conflict definition:
    # A triple (a, b, c) is "bad" if lcm(a,b) == lcm(a,c) == lcm(b,c).
    # Let's build a graph where nodes are numbers 1..128.
    # But the constraint is on triples, not pairs.
    # This is an "independent set in a 3-uniform hypergraph" problem.
    # Standard Max Independent Set solvers work on graphs.
    # We can transform this:
    # Two numbers a and b are "conflicting" if there exists a c such that (a,b,c) is a bad triple?
    # No, that's not right. If we pick a and b, we just can't pick ANY c that forms a bad triple with them.
    # So if we have a set S, it is valid iff for all distinct a,b,c in S, they don't form a bad triple.
    
    # Strategy:
    # 1. Identify all "bad triples" (a,b,c) in [1,128].
    # 2. We want a subset S maximizing |S| such that no bad triple is contained in S.
    # This is the Maximum 3-SAT (or rather, Max-Independent-Set-in-3-Hypergraph) problem.
    
    # Heuristic Approach:
    # Since N=128 is small, we can use a greedy approach with backtracking or local search.
    # Given the time limit, a robust greedy + local improvement is safer than full branch-and-bound which might explode.
    
    # Let's try a different angle:
    # What structures cause equal LCMs?
    # If a|b and a|c, lcm(a,b)=b, lcm(a,c)=c. For them to be equal, b=c (not distinct).
    # So in a bad triple, no element divides another?
    # If a|b, then lcm(a,b)=b. If lcm(a,c)=b and lcm(b,c)=b, then c|b.
    # So if a|b and c|b, then lcm(a,b)=b, lcm(b,c)=b. We need lcm(a,c)=b.
    # This happens if a and c are "complementary" divisors of b.
    # Example: b=6, a=2, c=3. lcm(2,6)=6, lcm(3,6)=6, lcm(2,3)=6. Bad triple.
    # So, any set containing a divisor a of b, a divisor c of b, and b itself, where lcm(a,c)=b, is bad.
    
    # This suggests that "dense" sets of divisors are dangerous.
    # Numbers with many divisors (like 120, 128, etc.) are likely to be in many bad triples.
    
    # Let's precompute for each pair (a,b) if there exists a c such that (a,b,c) is bad.
    # If yes, then a and b cannot both be in the solution IF we are forced to exclude c?
    # No. If we include a and b, we must exclude ALL c that form a bad triple with a and b.
    
    # Let's build a conflict graph for pairs?
    # If we select a and b, we "ban" a set of candidates C_ab.
    # This is a complex constraint.
    
    # Alternative: Greedy with lookahead.
    # 1. Start with all numbers.
    # 2. Find all bad triples.
    # 3. While there is a bad triple in the current set:
    #    - Pick a bad triple (a,b,c).
    #    - Remove the element from the triple that is "least useful" (e.g., part of the most other bad triples).
    #    - Update bad triples list.
    
    # This is a standard greedy algorithm for hitting set / independent set in hypergraphs.
    # It is not guaranteed to be optimal but should be fast and good.
    
    # To improve: Local Search.
    # After the greedy phase, try to swap elements or add removed elements if possible.
    
    # Let's implement the Greedy Hitting Set approach.
    
    # 1. Generate all bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            l_ab = lcm(a, b)
            # c must be > b to avoid duplicates in generation, but c can be any distinct.
            # Actually, just iterate all c > b.
            for c in range(b + 1, N + 1):
                l_ac = lcm(a, c)
                if l_ac != l_ab:
                    continue
                l_bc = lcm(b, c)
                if l_bc == l_ab:
                    bad_triples.append((a, b, c))
    
    # Map from element to list of indices of bad triples it belongs to
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    for i, t in enumerate(bad_triples):
        for x in t:
            elem_to_triples[x].append(i)
    
    # Current set of active elements
    active = set(range(1, N + 1))
    # Current set of active triple indices
    active_triples = set(range(len(bad_triples)))
    
    # Greedy removal
    # We want to remove elements to hit all active triples.
    # Minimize number of removals = Maximize remaining.
    # Greedy: Pick element involved in most active triples, remove it.
    
    # To make it better, we can randomize the choice among top candidates.
    # And run multiple restarts.
    
    best_set = set()
    best_size = 0
    
    def run_greedy(seed):
        random.seed(seed)
        active = set(range(1, N + 1))
        # We need to track which triples are still "unhit" (i.e., all 3 elements are active)
        # Initially all are unhit.
        # When we remove x, we hit all triples containing x.
        
        # Efficient tracking:
        # count[x] = number of active triples containing x
        # active_triples_count = total number of active triples
        
        counts = {x: len(elem_to_triples[x]) for x in active}
        
        # We need to know, for each triple, how many of its elements are still active.
        # If it drops to < 3, it is "hit" and no longer a constraint.
        # Let's maintain a list of active triples.
        
        # This naive update might be slow if we do it repeatedly.
        # But N=128, triples are likely < 10000.
        
        # Let's just maintain a set of active triples.
        curr_active_triples = set(range(len(bad_triples)))
        
        # Precompute triple elements for fast access
        # bad_triples is list of tuples
        
        while curr_active_triples:
            if time.time() > deadline:
                return active
            
            # Find element with max count in active triples
            # This requires iterating all active elements and checking their triples?
            # Or we can maintain a heap.
            
            # Simple O(N * T) check might be too slow if done many times.
            # Let's approximate: just pick a random active triple, and remove one of its elements.
            # Which one? The one with the highest degree (in active triples).
            
            # To be efficient, let's just pick a random active triple.
            t_idx = random.choice(list(curr_active_triples))
            t = bad_triples[t_idx]
            
            # Calculate current "weight" for each element in t
            # Weight = number of active triples containing this element
            # We can estimate this or compute it.
            # Computing exactly requires scanning all active triples for each x in t.
            # |t|=3. T is maybe ~1000. 3*1000 = 3000 ops.
            # If we do this ~100 times, 300,000 ops. Very fast.
            
            weights = []
            for x in t:
                w = 0
                for at_idx in curr_active_triples:
                    if x in bad_triples[at_idx]:
                        w += 1
                weights.append((w, x))
            
            # Remove the one with max weight
            weights.sort(reverse=True)
            x_to_remove = weights[0][1]
            
            active.remove(x_to_remove)
            
            # Update curr_active_triples: remove all triples containing x_to_remove
            to_remove = []
            for at_idx in curr_active_triples:
                if x_to_remove in bad_triples[at_idx]:
                    to_remove.append(at_idx)
            
            for at_idx in to_remove:
                curr_active_triples.remove(at_idx)
                
            if len(active) > best_size:
                best_size = len(active)
                best_set = set(active)
                
        return active

    # Run multiple restarts with different seeds
    # How many? 
    # Each run takes maybe 0.1s?
    # We have 50s. So 500 restarts?
    
    # Let's profile one run mentally.
    # Number of removals <= 128.
    # Each removal: 
    # 1. Pick random triple (O(1) amortized if we keep list, but set iteration is O(T)). 
    #    Converting set to list is O(T). 
    #    Better: keep a list of active triple indices, and filter?
    #    Or just use a list and swap-remove.
    
    # Let's optimize the inner loop.
    
    def run_greedy_fast(seed):
        random.seed(seed)
        active = set(range(1, N + 1))
        
        # List of active triple indices
        active_triple_list = list(range(len(bad_triples)))
        
        # To speed up "remove x from all triples", we can just filter the list.
        # But we need to pick a random triple from the current list.
        
        while active_triple_list:
            if time.time() > deadline:
                return active
            
            # Pick random triple
            idx = random.randint(0, len(active_triple_list) - 1)
            t = bad_triples[active_triple_list[idx]]
            
            # Calculate weights for x in t
            # Count how many times x appears in active_triple_list
            # This is O(T) for each x.
            # T is at most ~5000 (guess).
            # 3 * 5000 = 15000 ops.
            # 128 removals * 15000 = ~2M ops.
            # Python does ~10M-50M ops/sec. So 0.1s per run.
            # 500 runs = 50s. Fits budget.
            
            weights = []
            for x in t:
                # Optimization: precompute a counter?
                # No, active_triple_list changes.
                w = 0
                for at_idx in active_triple_list:
                    if x in bad_triples[at_idx]:
                        w += 1
                weights.append((w, x))
            
            weights.sort(reverse=True)
            x_to_remove = weights[0][1]
            
            active.remove(x_to_remove)
            
            # Filter active_triple_list
            # Keep only triples that do NOT contain x_to_remove
            active_triple_list = [at_idx for at_idx in active_triple_list 
                                  if x_to_remove not in bad_triples[at_idx]]
            
            if len(active) > best_size:
                best_size = len(active)
                best_set = set(active)
                
        return active

    # Execute restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 5: # 5s buffer
            break
        res = run_greedy_fast(i)
        # The global variables best_set and best_size are updated inside run_greedy_fast?
        # No, they are local to the function scope if not declared global.
        # I should pass them or return the best found in that run.
        # Let's refactor run_greedy_fast to return the best set found during that run.
        
    # Wait, the previous code used global best_set. Let's fix that.
    
    def run_greedy_fast_local(seed):
        random.seed(seed)
        active = set(range(1, N + 1))
        active_triple_list = list(range(len(bad_triples)))
        local_best = set(active)
        local_best_size = len(active)
        
        while active_triple_list:
            if time.time() > deadline:
                break
            
            idx = random.randint(0, len(active_triple_list) - 1)
            t = bad_triples[active_triple_list[idx]]
            
            weights = []
            for x in t:
                w = 0
                for at_idx in active_triple_list:
                    if x in bad_triples[at_idx]:
                        w += 1
                weights.append((w, x))
            
            weights.sort(reverse=True)
            x_to_remove = weights[0][1]
            
            active.remove(x_to_remove)
            
            active_triple_list = [at_idx for at_idx in active_triple_list 
                                  if x_to_remove not in bad_triples[at_idx]]
            
            if len(active) > local_best_size:
                local_best_size = len(active)
                local_best = set(active)
                
        return local_best

    # Run restarts
    global_best = set()
    global_best_size = 0
    
    for i in range(200): # Start with 200, adjust if time permits
        if time.time() > deadline - 5:
            break
        res = run_greedy_fast_local(i)
        if len(res) > global_best_size:
            global_best_size = len(res)
            global_best = res
            
    # If we have time, try local search on the best solution found?
    # Or just return the best greedy result.
    # The greedy result is a valid set.
    
    if not global_best:
        # Fallback: just return all numbers? No, must be valid.
        # The greedy always returns a valid set (empty or not).
        # If it removed everything, it's empty.
        # But we want non-empty.
        # If global_best is empty, pick a random number.
        global_best = {1}
        
    return list(global_best)

# EVOLVE-BLOCK-END
