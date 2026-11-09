import time
import sys
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # We need a Sidon set containing 0, with differences 1..30.
    # A Sidon set of size k has C(k,2) distinct differences.
    # We need to cover 1..30.
    # Let's search for a set of size k.
    # Known small Sidon sets with good coverage:
    # k=8: C(8,2)=28 diffs. Cannot cover 30.
    # k=9: C(9,2)=36 diffs. Possible.
    # k=10: C(10,2)=45 diffs. Easier.
    # We want to minimize the max element (diameter).
    
    # Strategy:
    # 1. Try to find a valid set of size k starting from k=9, then k=10, etc.
    # 2. For each k, use backtracking to find a set with minimal diameter.
    # 3. To guide the search, we can use the fact that differences 1..30 must be present.
    
    # We will use a backtracking solver that builds the set element by element.
    # To make it efficient, we will fix the first few elements and search for the rest.
    # Actually, a better approach for small k:
    # Generate candidate sets using a greedy heuristic with randomization, then refine.
    # But the prompt asks for a CSP/Backtracking approach.
    
    # Let's define the CSP:
    # Variables: x_0=0, x_1, ..., x_{k-1} such that 0 = x_0 < x_1 < ... < x_{k-1}.
    # Constraints:
    # 1. All differences x_j - x_i (j>i) are distinct.
    # 2. {1, ..., 30} is a subset of {x_j - x_i | j>i}.
    
    # Since we want to minimize max(x), we can do a binary search on D (diameter)
    # or just search for any valid set and then try to shrink it.
    # Given the small size, we can try to construct a set with small diameter directly.
    
    # Heuristic: 
    # Start with 0.
    # We need to cover 1..30.
    # Let's try to find a set of size 9 or 10.
    
    # A known perfect difference set of size q+1 (where q is prime power) has diameter q(q+1).
    # For q=8 (not prime power), q=9 (not prime power), q=11 (prime power? No, 11 is prime, so q=11 is prime power 11^1).
    # Wait, perfect difference sets exist for q = p^m - 1 where p is prime.
    # If q=8, q+1=9. 8 is not of form p^m-1? 2^3-1=7. 3^2-1=8. Yes!
    # So there is a perfect difference set of size 9 (q=8) with diameter 8*9 = 72?
    # Actually, Singer difference sets have size q+1 and diameter q(q+1).
    # For q=8, size 9, diameter 72.
    # Does it contain 1..30? Yes, perfect difference sets contain all differences 1..q(q+1).
    # So a set of size 9 with diameter 72 is a valid solution.
    # Can we do better?
    # Maybe a non-perfect Sidon set with smaller diameter?
    # Let's try to find a set of size 9 with diameter < 72.
    # Or size 10 with diameter < 72?
    
    # Let's implement a backtracking search for a set of size k with diameter <= D.
    # We will iterate D from a lower bound upwards.
    # Lower bound: We need 30 differences.
    # If we have k elements, we have k(k-1)/2 differences.
    # To cover 1..30, we need at least 30 differences.
    # k=8 -> 28 diffs (impossible).
    # k=9 -> 36 diffs.
    # k=10 -> 45 diffs.
    
    # Let's try k=9 first.
    # We can use a randomized backtracking to find a solution, then try to minimize D.
    
    def solve_csp(k, max_diameter, seed=0):
        """
        Find a Sidon set of size k, containing 0, with max element <= max_diameter,
        covering all differences 1..30.
        Returns the set or None.
        """
        random.seed(seed)
        
        # We will build the set incrementally.
        # x[0] = 0
        # We need to choose x[1] ... x[k-1]
        
        # To speed up, we can precompute which differences are "critical" (1..30).
        
        # Backtracking state:
        # current_set: list of chosen elements
        # used_diffs: set of all differences generated so far
        # covered_diffs: set of differences in 1..30 generated so far
        
        # Heuristic: Choose the next element x_i.
        # To minimize diameter, we should probably pick smaller numbers first?
        # But we need to ensure we can still cover the remaining differences.
        
        # A simpler approach for small k:
        # Iterate through all possible combinations? No, 100^8 is too big.
        
        # Let's use a recursive backtracking with pruning.
        
        # To make it efficient, we can fix the first few elements.
        # For k=9, let's try to find a solution.
        
        # We will use a "randomized restart" strategy with backtracking.
        
        def backtrack(idx, current_set, used_diffs, covered_diffs):
            if time.time() > deadline:
                return None
            
            if idx == k:
                if len(covered_diffs) == 30:
                    return current_set[:]
                return None
            
            # MRV: Which value to try next?
            # We want to keep the set small.
            # Let's try values from current_set[-1] + 1 up to max_diameter.
            
            # Optimization: If we are near the end, we can check if the remaining
            # capacity can cover the missing differences.
            
            # For simplicity, we'll just iterate.
            # To improve performance, we can randomize the order of trying values
            # or use a heuristic.
            
            # Let's try a range of values.
            # Start from current_set[-1] + 1
            # To avoid huge search spaces, we might want to restrict the range
            # or use randomization.
            
            # Let's try a deterministic sweep first, then randomize if stuck.
            
            # Actually, for k=9, the space is small enough if we prune well.
            # Pruning:
            # 1. New differences must not conflict with used_diffs.
            # 2. We must be able to cover the remaining differences with the remaining elements.
            
            # Let's implement a basic backtracking.
            
            # To avoid infinite loops or huge trees, we limit the search depth/time.
            
            # We will try to pick the next element x.
            # Range: [current_set[-1] + 1, max_diameter]
            
            # To make it faster, we can try to pick x such that it generates
            # many new differences in 1..30.
            
            # Let's generate candidates.
            # For each candidate x:
            #   new_diffs = {x - y for y in current_set}
            #   if any d in new_diffs is in used_diffs: conflict
            #   else: recurse
            
            # To speed up, we can pre-filter candidates.
            
            # Let's just do a simple DFS with random restarts.
            
            # Since we are inside a function that might be called multiple times,
            # we need to be careful with state.
            
            # We'll use a global counter for nodes visited to abort if too slow.
            
            pass # Placeholder
            
        # Let's implement a more direct search:
        # We know a perfect difference set of size 9 exists with diameter 72.
        # Let's try to find one.
        
        # We can construct it using the Singer construction if we know the field,
        # but that's complex.
        
        # Alternative: Use the known set for q=8.
        # The differences are 1..72.
        # One such set is derived from the projective plane of order 8?
        # Actually, let's just search.
        
        # We will use a randomized local search (Simulated Annealing) but with a 
        # much better initial guess and faster evaluation.
        # The prompt asks for CSP/Backtracking, but if SA is faster and finds
        # the same result, is it better? The prompt says "Strategy to try in this change: 
        # Model the problem as a Constraint Satisfaction Problem (CSP) using backtracking...".
        
        # So I must implement backtracking.
        
        # To make backtracking efficient for k=9:
        # We can fix x0=0.
        # We need to find x1..x8.
        # We can use the fact that the differences must be unique.
        
        # Let's try a different angle:
        # We know the answer is likely close to 72.
        # We can do a binary search on D (diameter) from 50 to 100.
        # For each D, we run a backtracking solver.
        
        # Backtracking Solver:
        # State: (idx, current_set, used_diffs, covered_diffs)
        # Heuristic: 
        #   - Try to add elements that cover many missing diffs.
        #   - Keep elements small to allow room for future elements.
        
        # Let's implement a recursive solver with memoization? 
        # State space is too large for memoization on the set itself.
        
        # We will use a "dancing links" style or simple recursion.
        
        def find_solution(k, D, seed):
            random.seed(seed)
            nodes_visited = 0
            max_nodes = 1000000 # Abort if too many nodes
            
            def dfs(idx, curr, used_diffs, covered):
                nonlocal nodes_visited
                if time.time() > deadline:
                    return None
                if nodes_visited > max_nodes:
                    return None
                
                if idx == k:
                    if len(covered) == 30:
                        return curr[:]
                    return None
                
                # Determine next value
                # We want to choose x such that:
                # 1. x > curr[-1]
                # 2. x <= D
                # 3. No difference conflicts
                
                # To guide the search, let's prioritize x that covers missing diffs.
                # But we also need to leave room.
                
                # Let's try values from curr[-1]+1 to D.
                # To avoid linear scan, we can randomize or use a heuristic.
                
                # Heuristic: 
                # Calculate for each candidate x the number of new diffs in 1..30.
                # Pick the one with the highest count?
                
                # This is expensive. Let's just do a randomized DFS.
                
                # Generate candidates
                start = curr[-1] + 1
                end = D
                if start > end:
                    return None
                
                # To make it faster, we can skip values that definitely conflict.
                
                # Let's try a subset of candidates to reduce branching.
                # For k=9, branching factor D is ~70.
                # 70^8 is too big.
                
                # We need strong pruning.
                # Pruning: 
                # If the number of remaining elements is r, the max number of new diffs is r*(k-idx).
                # We need to cover 30 - len(covered) diffs.
                # If r*(k-idx) < 30 - len(covered), prune.
                
                # Also, the max possible diameter is D.
                
                # Let's implement the DFS.
                
                # To avoid deep recursion limits, we might need to increase recursion limit,
                # but k=9 is small.
                
                # We will try candidates in a randomized order to escape local minima.
                
                candidates = list(range(start, end + 1))
                random.shuffle(candidates)
                
                for x in candidates:
                    nodes_visited += 1
                    if nodes_visited % 10000 == 0 and time.time() > deadline:
                        return None
                    
                    # Check conflicts
                    conflict = False
                    new_diffs = []
                    for y in curr:
                        d = x - y
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.append(d)
                    
                    if conflict:
                        continue
                    
                    new_used = used_diffs | set(new_diffs)
                    new_covered = covered | (set(new_diffs) & set(range(1, 31)))
                    
                    # Pruning: Can we still cover the missing diffs?
                    remaining_elements = k - idx - 1
                    max_new_diffs = remaining_elements * (k - idx) # Each new element adds at most (current size) diffs
                    # Actually, new element adds (idx+1) diffs.
                    # Next adds (idx+2), etc.
                    # Total new diffs possible = sum_{j=1}^{remaining_elements} (idx + j)
                    # = remaining_elements * idx + remaining_elements*(remaining_elements+1)/2
                    
                    missing = 30 - len(new_covered)
                    if missing > 0:
                        # Calculate max possible new covered diffs
                        # This is an upper bound.
                        pass # Skipping complex bound for now, rely on search
                    
                    res = dfs(idx + 1, curr + [x], new_used, new_covered)
                    if res:
                        return res
                
                return None
            
            return dfs(1, [0], set(), set())
        
        # We will try to find a solution for k=9 with increasing D.
        # Start D from 60 (lower bound estimate) to 100.
        
        best_set = None
        best_D = float('inf')
        
        # Try k=9
        for D in range(60, 101):
            # Try a few random seeds for each D
            for seed in range(5):
                sol = find_solution(9, D, seed)
                if sol:
                    best_set = sol
                    best_D = D
                    break
            if best_set:
                break
        
        # If k=9 didn't work or took too long, try k=10
        if not best_set:
            for D in range(50, 101):
                for seed in range(5):
                    sol = find_solution(10, D, seed)
                    if sol:
                        best_set = sol
                        best_D = D
                        break
                if best_set:
                    break
        
        if best_set:
            return best_set
        
        # Fallback: Use a known valid set if search fails
        # Example: A known Sidon set with differences 1..30
        # Let's construct one manually or use a default.
        # Default: [0, 1, 4, 10, 12, 17, 21, 25, 29, 34, 41, 45, 52, 58, 63, 68, 73, 79, 84, 90, 95, 101, 107, 112, 118, 124, 129, 135, 141, 146, 152, 158]
        # This is too large.
        
        # Let's try a greedy construction as fallback.
        def greedy_construct():
            a = [0]
            def get_diffs(s):
                return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
            for d in range(1, 31):
                if d in get_diffs(sorted(a)):
                    continue
                found = False
                for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                    trial = sorted(a + [x])
                    if len(set(get_diffs(trial))) == len(get_diffs(trial)):
                        a = trial
                        found = True
                        break
                if not found:
                    y = 2 * max(a) + 3 * d + 1
                    while True:
                        trial = sorted(a + [y, y + d])
                        if len(set(get_diffs(trial))) == len(get_diffs(trial)):
                            a = trial
                            break
                        y += 1
            return a
        
        return greedy_construct()

# EVOLVE-BLOCK-END
