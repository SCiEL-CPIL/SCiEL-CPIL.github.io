import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin under 90s
    
    N = 256
    
    # Precompute prime factors for 1..N
    # F[x] is a frozenset of prime factors of x
    F = [None] * (N + 1)
    F[1] = frozenset()
    for i in range(2, N + 1):
        x = i
        factors = set()
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.add(d)
                while x % d == 0:
                    x //= d
            else:
                d += 1
        if x > 1:
            factors.add(x)
        F[i] = frozenset(factors)
    
    def greedy_initialization(descending=True, randomize=False):
        candidates = list(range(1, N + 1))
        if descending:
            candidates.sort(reverse=True)
        else:
            candidates.sort()
        if randomize:
            random.shuffle(candidates)
        
        selected = []
        selected_factors = set()
        
        for a in candidates:
            if time.time() > deadline:
                break
            fa = F[a]
            if selected_factors.isdisjoint(fa):
                selected.append(a)
                selected_factors.update(fa)
        
        return selected

    def solve_subproblem(candidates, used_primes=None, time_budget=5.0):
        """
        Find max sum subset of candidates that is pairwise coprime AND coprime to used_primes.
        candidates: list of ints
        used_primes: set of primes already used (must be disjoint)
        Returns: (max_sum, subset_list)
        """
        if not candidates:
            return 0, []
        
        if used_primes is None:
            used_primes = set()
        
        # Filter candidates that are coprime to used_primes
        cands = [c for c in candidates if F[c].isdisjoint(used_primes)]
        if not cands:
            return 0, []
        
        # Sort descending to find high sums early
        cands.sort(reverse=True)
        n = len(cands)
        
        # If small, do exact search with better pruning
        if n <= 25:
            suffix_sums = [0] * (n + 1)
            for i in range(n - 1, -1, -1):
                suffix_sums[i] = suffix_sums[i + 1] + cands[i]
            
            best_sum = 0
            best_subset = []
            
            def dfs(idx, current_sum, cur_primes, current_subset):
                nonlocal best_sum, best_subset
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_subset = current_subset[:]
                
                if idx == n:
                    return
                
                # Pruning
                if current_sum + suffix_sums[idx] <= best_sum:
                    return
                
                # Option 1: Exclude
                dfs(idx + 1, current_sum, cur_primes, current_subset)
                
                # Option 2: Include if coprime
                fa = F[cands[idx]]
                if cur_primes.isdisjoint(fa):
                    current_subset.append(cands[idx])
                    new_primes = cur_primes.union(fa)
                    dfs(idx + 1, current_sum + cands[idx], new_primes, current_subset)
                    current_subset.pop()
            
            dfs(0, 0, set(), [])
            return best_sum, best_subset
        else:
            # For larger problems, use greedy with local improvement
            selected = []
            selected_factors = set()
            for a in cands:
                if F[a].isdisjoint(selected_factors):
                    selected.append(a)
                    selected_factors.update(F[a])
            
            # Try to improve with simple local search on the greedy solution
            improved = True
            while improved and time.time() < start_time + time_budget:
                improved = False
                sel_set = set(selected)
                for a in list(selected):
                    rest = sel_set - {a}
                    rest_factors = set()
                    for x in rest:
                        rest_factors.update(F[x])
                    # Try to find a better replacement
                    for b in cands:
                        if b in rest:
                            continue
                        if F[b].isdisjoint(rest_factors) and b > a:
                            new_sel = rest | {b}
                            if sum(new_sel) > sum(selected):
                                selected = list(new_sel)
                                improved = True
                                break
                    if improved:
                        break
            return sum(selected), selected

    def local_search(S, max_iterations=100):
        """
        Perform local search with 1-exchange and 2-exchange moves.
        S: set of ints
        """
        S = set(S)
        best_sum = sum(S)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            if time.time() > deadline - 2:
                break
            
            # Compute factors of S
            S_factors = set()
            for x in S:
                S_factors.update(F[x])
            
            # Try 1-exchange: remove a, add best subset from remaining candidates
            for a in list(S):
                if time.time() > deadline - 2:
                    return S
                
                rest = S - {a}
                rest_factors = set()
                for x in rest:
                    rest_factors.update(F[x])
                
                # Find all candidates coprime to rest_factors
                candidates = [x for x in range(1, N + 1) if x not in rest and F[x].isdisjoint(rest_factors)]
                
                # Solve subproblem: find best subset T of candidates
                # We want sum(T) > a for improvement
                if candidates:
                    best_T_sum, best_T = solve_subproblem(candidates, used_primes=rest_factors, time_budget=3.0)
                    if best_T_sum > a:
                        S = rest.union(best_T)
                        best_sum = sum(S)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try 2-exchange: remove two elements a,b, add best subset
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline - 2:
                    return S
                for j in range(i + 1, len(S_list)):
                    a, b = S_list[i], S_list[j]
                    rest = S - {a, b}
                    rest_factors = set()
                    for x in rest:
                        rest_factors.update(F[x])
                    
                    candidates = [x for x in range(1, N + 1) if x not in rest and F[x].isdisjoint(rest_factors)]
                    
                    if candidates:
                        best_T_sum, best_T = solve_subproblem(candidates, used_primes=rest_factors, time_budget=2.0)
                        if best_T_sum > a + b:
                            S = rest.union(best_T)
                            best_sum = sum(S)
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try adding an element without removing (if there's slack)
            added = False
            for x in range(1, N + 1):
                if x in S:
                    continue
                if F[x].isdisjoint(S_factors):
                    S.add(x)
                    S_factors.update(F[x])
                    best_sum = sum(S)
                    improved = True
                    added = True
                    break
            
            if added:
                continue
        
        return S

    best_S = set()
    best_sum = 0
    
    # 1. Greedy Descending
    S1 = greedy_initialization(descending=True)
    S1_set = set(S1)
    sum1 = sum(S1_set)
    if sum1 > best_sum:
        best_S = S1_set
        best_sum = sum1
        
    # 2. Greedy Ascending
    S2 = greedy_initialization(descending=False)
    S2_set = set(S2)
    sum2 = sum(S2_set)
    if sum2 > best_sum:
        best_S = S2_set
        best_sum = sum2

    # 3. Randomized Greedy
    random.seed(42)
    for i in range(15):
        if time.time() > deadline - 10:
            break
        Sr = greedy_initialization(descending=True, randomize=True)
        Sr_set = set(Sr)
        sumr = sum(Sr_set)
        if sumr > best_sum:
            best_S = Sr_set
            best_sum = sumr

    # 4. Local Search on Best
    if time.time() < deadline - 5:
        best_S = local_search(best_S)
        best_sum = sum(best_S)

    # 5. If time remains, try local search on other initializations
    if time.time() < deadline - 5:
        for i in range(12):
            if time.time() > deadline - 2:
                break
            Sr = greedy_initialization(descending=True, randomize=True)
            Sr_set = local_search(set(Sr))
            sumr = sum(Sr_set)
            if sumr > best_sum:
                best_S = Sr_set
                best_sum = sumr

    # Final: try adding any missing coprime elements
    S_factors = set()
    for x in best_S:
        S_factors.update(F[x])
    for x in range(1, N + 1):
        if x not in best_S and F[x].isdisjoint(S_factors):
            best_S.add(x)
            S_factors.update(F[x])
    
    # Ensure return type is list of ints
    return [int(x) for x in best_S]
