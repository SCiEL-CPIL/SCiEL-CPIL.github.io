import random
import time
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe within 60s/90s
    
    N = 128
    # Precompute divisors for each number 1..N
    # divisors[x] will contain all d > 1 such that d divides x
    divisors = [[] for _ in range(N + 1)]
    for d in range(2, N + 1):
        for multiple in range(d, N + 1, d):
            divisors[multiple].append(d)
    
    def is_valid_addition(x, counts):
        """Check if x can be added to the current set given the counts of multiples for each d."""
        for d in divisors[x]:
            if counts[d] >= 2:
                return False
        return True

    def add_element(x, counts):
        for d in divisors[x]:
            counts[d] += 1

    def remove_element(x, counts):
        for d in divisors[x]:
            counts[d] -= 1

    def greedy_construct():
        counts = defaultdict(int)
        S = set()
        # Order can matter. Try increasing order first.
        for x in range(1, N + 1):
            if is_valid_addition(x, counts):
                S.add(x)
                add_element(x, counts)
        return S, counts

    def local_search(S, counts):
        """
        Perform local search to improve the size of S.
        Moves:
        1. Try to add a random element not in S.
        2. Try to swap an element in S with one not in S (if adding the new one requires removing the old one? 
           Actually, just try adding. If it fails, try swapping.
           A swap (remove y, add x) is valid if after removing y, x can be added.
        """
        current_S = set(S)
        current_counts = defaultdict(int, counts)
        best_size = len(current_S)
        
        # List of candidates not in S
        candidates = [x for x in range(1, N + 1) if x not in current_S]
        
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                break
            
            random.shuffle(candidates)
            improved = False
            
            # Try adding candidates
            for x in candidates:
                if is_valid_addition(x, current_counts):
                    current_S.add(x)
                    add_element(x, current_counts)
                    candidates.remove(x)
                    best_size = len(current_S)
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue
                
            # If no simple add worked, try swaps
            # Remove a random element y, then try to add x
            # To be efficient, we can pick a random y from S and a random x from candidates
            # But checking all pairs is O(N^2). N=128, so 128*128 ~ 16k checks. Fast enough.
            
            S_list = list(current_S)
            if not S_list or not candidates:
                break
                
            # Try a few random swaps
            swap_attempted = False
            for _ in range(50): # Limit swap attempts per iteration to avoid getting stuck
                if time.time() > deadline:
                    return current_S, current_counts
                
                y = random.choice(S_list)
                x = random.choice(candidates)
                
                # Check if swapping y for x is valid
                # Temporarily remove y
                remove_element(y, current_counts)
                
                if is_valid_addition(x, current_counts):
                    # Valid swap
                    current_S.remove(y)
                    current_S.add(x)
                    add_element(x, current_counts)
                    
                    # Update candidates list
                    # y is now a candidate, x is no longer a candidate
                    if x in candidates:
                        candidates.remove(x)
                    candidates.append(y)
                    
                    best_size = len(current_S)
                    no_improve_count = 0
                    improved = True
                    break
                else:
                    # Restore y
                    add_element(y, current_counts)
            
            if not improved:
                no_improve_count += 1
                # If stuck, maybe restart or accept a worse move? 
                # For now, just break to restart if no improvement.
                if no_improve_count > 50:
                    break

        return current_S, current_counts

    best_S = set()
    best_counts = defaultdict(int)
    
    # Run multiple restarts with different initializations or just one good greedy + LS
    # Since greedy is deterministic, we can try different orderings for greedy
    # Or just run LS on the greedy result.
    
    # 1. Standard Greedy
    S_greedy, counts_greedy = greedy_construct()
    S_ls, counts_ls = local_search(S_greedy, counts_greedy)
    
    if len(S_ls) > len(best_S):
        best_S = S_ls
        best_counts = counts_ls

    # 2. Try greedy with random shuffles
    for _ in range(20):
        if time.time() > deadline - 5:
            break
            
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        counts = defaultdict(int)
        S = set()
        for x in order:
            if is_valid_addition(x, counts):
                S.add(x)
                add_element(x, counts)
        
        S_ls, counts_ls = local_search(S, counts)
        
        if len(S_ls) > len(best_S):
            best_S = S_ls
            best_counts = counts_ls

    return list(best_S)

# Keep the function name as requested
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed program: def construct(): ... return selected

# We need to ensure we return a non-empty subset.
# The greedy will always include 1, and likely many others.

# Final check for validity before returning?
# The local search ensures validity at each step, so the result should be valid.
# However, it's good practice to verify.

def _verify(S):
    S = list(S)
    from itertools import combinations
    for a, b, c in combinations(S, 3):
        g1 = gcd(a, b)
        g2 = gcd(a, c)
        g3 = gcd(b, c)
        if g1 == g2 == g3:
            return False
    return True

# Wrap construct to include verification and fallback if needed (though LS should be safe)
_original_construct = construct

def construct():
    res = _original_construct()
    if not _verify(res):
        # Fallback to simple greedy if something went wrong in LS
        # This should rarely happen
        from math import gcd
        from itertools import combinations
        selected = []
        for x in range(1, 129):
            if all(len({gcd(a,b),gcd(a,x),gcd(b,x)}) > 1 for a,b in combinations(selected,2)):
                selected.append(x)
        return selected
    return res
