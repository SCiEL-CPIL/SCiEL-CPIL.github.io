import time
import random

def _build_dp(elements):
    """Build DP bitsets for subset sums by cardinality."""
    n = len(elements)
    if n == 0:
        return [0] * 1
    dp = [0] * (n + 1)
    dp[0] = 1
    for e in elements:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k-1] << e)
    return dp

def _is_valid_mean_free_with_dp(A, x):
    """
    Check if adding x to A keeps the set mean-free.
    A is a list of distinct integers.
    Returns True if A U {x} is valid.
    """
    n = len(A)
    
    # 1. Check if x is the mean of any subset of A of size k >= 2
    if n >= 2:
        dp_A = _build_dp(A)
        for k in range(2, n + 1):
            target = k * x
            if target < 0:
                continue
            if (dp_A[k] >> target) & 1:
                return False
    # If n < 2, x cannot be mean of subset of A of size >= 2.
    
    # 2. Check if any a in A is the mean of any subset of (A U {x}) \ {a} of size k >= 2
    # Such a subset must contain x (since subsets of A \ {a} were already valid in A)
    # Let S be the subset. S contains x.
    # S = T U {x}, T subset of A \ {a}.
    # |S| = k. |T| = k - 1.
    # Sum(S) = Sum(T) + x = k * a.
    # Sum(T) = k * a - x.
    # We need to check if there exists T subset of A \ {a} with |T| = k - 1 and Sum(T) = k * a - x.
    
    # Optimization: Pre-compute sum of A?
    # We iterate a in A.
    for a in A:
        others = [v for v in A if v != a]
        m = len(others)
        if m < 1:
            continue # T must have size k-1 >= 1 for k>=2, so others must not be empty if k>1. 
                     # Actually if m=0, T is empty, k=1, but k>=2 required. So no subset.
        
        # Build DP for others
        dp_others = _build_dp(others)
        
        # k ranges from 2 to len(A) (since |S| <= len(A) as x is added to A of size n, total size n+1, S is subset of size n, so k <= n)
        # Wait, S is subset of (A U {x}) \ {a}. Size of (A U {x}) \ {a} is n.
        # So k can be at most n.
        # |T| = k - 1. Max |T| is n - 1.
        # dp_others has indices up to len(others) = n - 1.
        
        for k in range(2, n + 1):
            target_sum = k * a - x
            if target_sum < 0:
                continue
            j = k - 1
            if j > len(others):
                continue
            # Check if bit target_sum is set in dp_others[j]
            if (dp_others[j] >> target_sum) & 1:
                return False
                
    return True

def _is_valid_set(S):
    """Check if S is a valid mean-free set."""
    for i, x in enumerate(S):
        others = [v for v in S if v != x]
        if not _is_valid_mean_free_with_dp(others, x):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    best_set = []
    best_len = 0
    
    # Strategy: Local Search
    # 1. Generate initial set via Greedy
    # 2. Improve via Swaps and Additions
    
    # Helper to get a random valid set using greedy
    def greedy_generate(seed_order=None):
        random.seed(seed_order) if seed_order is not None else None
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        current_set = []
        for x in candidates:
            if time.time() > deadline:
                break
            if _is_valid_mean_free_with_dp(current_set, x):
                current_set.append(x)
        return current_set

    # Run a few greedy restarts to find a good starting point
    for i in range(10):
        if time.time() > deadline:
            break
        S = greedy_generate(42 + i)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S[:]
        
        # Local Search on S
        # We try to improve S
        # Time budget per local search: ~5-8 seconds?
        ls_start = time.time()
        ls_deadline = ls_start + 5.0
        
        no_improvement_count = 0
        max_no_improvement = 50
        
        while time.time() < ls_deadline:
            improved = False
            
            # Try to add a random element
            # We need to pick x not in S
            # To be efficient, we can pick a few random candidates
            for _ in range(20):
                if time.time() > ls_deadline:
                    break
                x = random.randint(1, 512)
                if x in S:
                    continue
                if _is_valid_mean_free_with_dp(S, x):
                    S.append(x)
                    improved = True
                    no_improvement_count = 0
                    break
            
            if improved:
                continue
            
            # If no addition, try a swap
            # Pick random a in S, random x not in S
            # Check if (S \ {a}) U {x} is valid
            # Note: This is expensive.
            # We can optimize by only trying swaps if we've been stuck for a while
            if no_improvement_count > 5:
                for _ in range(10):
                    if time.time() > ls_deadline:
                        break
                    a = random.choice(S)
                    x = random.randint(1, 512)
                    if x in S:
                        continue
                    
                    # Check validity of S' = (S \ {a}) U {x}
                    # This is equivalent to checking if x is valid to add to (S \ {a})
                    # AND that a is not mean of subset of (S \ {a}) U {x} ? 
                    # Wait.
                    # The condition is that S' is a valid set.
                    # S' = S_new.
                    # We know S was valid.
                    # We remove a, add x.
                    # We need to check:
                    # 1. x is not mean of any subset of S \ {a}.
                    # 2. No b in S \ {a} is mean of any subset of (S \ {a} \ {b}) U {x}.
                    # 3. a is not mean of any subset of (S \ {a}) U {x} \ {a} ? 
                    #    Wait, a is removed. So a is not in S'. So we don't check a.
                    #    We only check elements in S'.
                    #    Elements in S' are (S \ {a}) U {x}.
                    #    So we check x and all b in S \ {a}.
                    
                    # This is exactly _is_valid_mean_free_with_dp(S \ {a}, x)
                    # BUT _is_valid_mean_free_with_dp(A, x) checks:
                    # 1. x not mean of subset of A.
                    # 2. No a' in A is mean of subset of (A U {x}) \ {a'}.
                    # This covers all elements in A U {x}.
                    # So yes, we just need to check if x is valid to add to S \ {a}.
                    
                    others = [v for v in S if v != a]
                    if _is_valid_mean_free_with_dp(others, x):
                        # Perform swap
                        S.remove(a)
                        S.append(x)
                        improved = True
                        no_improvement_count = 0
                        break
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    break # Stuck in local minimum, break to restart or try different strategy
        
        if len(S) > best_len:
            best_len = len(S)
            best_set = S[:]
            
        if best_len >= 48:
            break
            
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(result[:10], "...", result[-10:])
