import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = 192
    max_k = 48
    
    # Precompute all triples of distinct elements from [1..n]
    # For each element x, precompute which triples (a,b,c) have x | (a+b+c)
    # This is expensive for n=192, so we'll compute on-the-fly but cache
    
    # Better approach: for each candidate x, compute forbidden pairs
    # x divides a+b+c means a+b+c = x, 2x, 3x, ... 
    # For a given x and a, the forbidden (b,c) pairs are those where b+c = kx - a for k>=1
    
    # Let's use a different structural approach:
    # Model as: select subset S of [1..n] such that for all x in S, 
    # for all distinct a,b,c in S\{x}, x does not divide a+b+c
    
    # Precompute: for each x in 1..n, for each pair (a,b) with a != b, a != x, b != x,
    # check if there exists c != x, c != a, c != b such that x | (a+b+c)
    # This is too expensive to precompute fully.
    
    # Instead, use a smarter approach:
    # 1. Start with a good initial solution from greedy
    # 2. Use local search with swap operations
    # 3. Use constraint propagation to prune
    
    def can_add(x, S):
        """Check if x can be added to set S (represented as a set for O(1) lookup)."""
        S_set = S if isinstance(S, set) else set(S)
        S_list = list(S_set)
        n_s = len(S_list)
        
        # 1. x divides sum of any 3 distinct elements in S?
        if n_s >= 3:
            for i in range(n_s):
                for j in range(i+1, n_s):
                    for k in range(j+1, n_s):
                        if (S_list[i] + S_list[j] + S_list[k]) % x == 0:
                            return False
        
        # 2. No element a in S divides x + u + v for any distinct u, v in S\{a}.
        if n_s >= 2:
            for i, a in enumerate(S_list):
                for j in range(n_s):
                    if j == i:
                        continue
                    u = S_list[j]
                    for k in range(j + 1, n_s):
                        if k == i:
                            continue
                        v = S_list[k]
                        if (x + u + v) % a == 0:
                            return False
        
        return True
    
    def run_greedy(order_type):
        """Run a greedy construction with a specific ordering."""
        candidates = list(range(1, n + 1))
        
        if order_type == 'desc':
            candidates.sort(reverse=True)
        elif order_type == 'asc':
            candidates.sort()
        elif order_type == 'random':
            random.shuffle(candidates)
        
        current_set = []
        current_set_set = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, current_set_set):
                current_set.append(x)
                current_set_set.add(x)
                if len(current_set) >= max_k:
                    break
        
        return current_set
    
    # First, try deterministic Descending
    best_set = run_greedy('desc')
    best_size = len(best_set)
    
    # Random restarts
    for i in range(20):
        if time.time() > deadline:
            break
        s = run_greedy('random')
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
    
    # Local search: try to add any remaining number
    if time.time() < deadline:
        current_best = set(best_set)
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if x not in current_best:
                    if can_add(x, current_best):
                        current_best.add(x)
                        added_any = True
                        break
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_set = list(current_best)
    
    # Structural change: Use a constraint-based backtracking with smart ordering
    # Key idea: process elements in order of "constrainedness" (how many conflicts they have)
    if time.time() < deadline - 10:
        # Compute conflict scores: for each x, count how many other elements it conflicts with
        # This helps us order the search
        conflict_scores = {}
        for x in range(1, n + 1):
            conflicts = 0
            for y in range(1, n + 1):
                if x == y:
                    continue
                # Check if x and y can coexist (simplified: just check pairwise)
                # This is a heuristic score
                if x % y == 0 or y % x == 0:
                    conflicts += 1
            conflict_scores[x] = conflicts
        
        # Sort candidates by conflict score (most constrained first)
        sorted_candidates = sorted(range(1, n + 1), key=lambda x: conflict_scores[x], reverse=True)
        
        # Backtracking search with pruning
        current = set(best_set)
        
        def backtrack(remaining_candidates, depth=0):
            nonlocal best_set, best_size
            if time.time() > deadline - 2:
                return
            if len(current) >= max_k:
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = list(current)
                return
            
            # Pruning: if we can't beat best_size even adding all remaining
            if len(current) + len(remaining_candidates) <= best_size:
                return
            
            if not remaining_candidates:
                return
            
            # Choose the next candidate to try
            # Heuristic: try candidates that conflict least with current set
            best_candidate = None
            best_conflict_count = float('inf')
            for x in remaining_candidates[:20]:  # Limit branching factor
                if time.time() > deadline - 2:
                    return
                if x in current:
                    continue
                # Count conflicts with current set
                conflict_count = 0
                for y in current:
                    if x != y:
                        # Simplified conflict check
                        if (x + y) % min(x, y) == 0:
                            conflict_count += 1
                if conflict_count < best_conflict_count:
                    best_conflict_count = conflict_count
                    best_candidate = x
            
            if best_candidate is None:
                return
            
            # Try adding best_candidate
            x = best_candidate
            if can_add(x, current):
                current.add(x)
                remaining = [c for c in remaining_candidates if c != x]
                backtrack(remaining, depth + 1)
                current.remove(x)
            
            # Try skipping best_candidate
            remaining = [c for c in remaining_candidates if c != x]
            backtrack(remaining, depth + 1)
        
        backtrack(sorted_candidates)
    
    return best_set
