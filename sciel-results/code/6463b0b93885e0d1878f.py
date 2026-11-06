import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin

    best_set = []
    best_len = 0
    
    # Helper to check validity quickly for small k
    def _fast_check_small_k(A, x):
        """
        Check if x is mean of subset of A of size 2 or 3.
        And if any a in A is mean of subset of (A U {x})\{a} of size 2 or 3 involving x.
        Returns False if invalid, True if possibly valid (needs full check).
        """
        n = len(A)
        if n == 0:
            return True
        if n == 1:
            a = A[0]
            # x is not mean of subset of A (size 1 < 2)
            # a is not mean of subset of {x} (size 1 < 2)
            return True
            
        # 1. x is mean of subset of A of size 2: 2x = a+b
        #    x is mean of subset of A of size 3: 3x = a+b+c
        
        # 2. a is mean of subset of (A U {x})\{a} of size 2 involving x:
        #    2a = x + b => b = 2a - x. b must be in A \ {a}.
        #    a is mean of subset of (A U {x})\{a} of size 3 involving x:
        #    3a = x + b + c => b+c = 3a - x. b,c in A \ {a}, b!=c.
        
        # We can check these using sets for O(1) or O(N) lookups.
        set_A = set(A)
        
        # Check 1: x mean of A (k=2,3)
        # k=2: exists a,b in A, a!=b, a+b=2x
        # We can just iterate pairs? No, N is small.
        # But we can use a set of sums? 
        # For fast check, let's just iterate.
        
        # Optimization: Pre-compute pair sums and triple sums of A?
        # If we do this every call, it's O(N^2) and O(N^3).
        # N is ~40. 40^3 = 64000. 1000 calls/sec = 64M ops. Might be tight.
        
        # Let's just use the full DP check but with early exit in the DP construction?
        # No, the DP construction is the cost.
        
        # Actually, the current `_is_valid_mean_free_with_dp` is the bottleneck.
        # Let's stick to the current DP logic but optimize the Python loops.
        
        return True # Fallback to full check

    def _is_valid_mean_free_with_dp(A, x):
        n = len(A)
        
        # 1. Check if x is the mean of any subset of A of size k >= 2
        if n >= 2:
            dp_A = [0] * (n + 1)
            dp_A[0] = 1
            for e in A:
                for k in range(n, 0, -1):
                    dp_A[k] |= (dp_A[k-1] << e)
            
            for k in range(2, n + 1):
                target = k * x
                if (dp_A[k] >> target) & 1:
                    return False
        
        # 2. Check if any a in A is the mean of any subset of (A U {x}) \ {a} of size k >= 2
        # Subset must contain x.
        # Sum(T) + x = k * a, T subset of A \ {a}, |T| = k - 1 = j.
        # Sum(T) = k * a - x.
        
        for a in A:
            others = [v for v in A if v != a]
            m = len(others)
            if m < 1:
                continue
            
            dp_others = [0] * (m + 1)
            dp_others[0] = 1
            for e in others:
                for k in range(m, 0, -1):
                    dp_others[k] |= (dp_others[k-1] << e)
            
            for k in range(2, n + 1):
                target_sum = k * a - x
                if target_sum < 0:
                    continue
                j = k - 1
                if j > m:
                    continue
                if (dp_others[j] >> target_sum) & 1:
                    return False
                
        return True

    def greedy_generate(seed_order=None):
        if seed_order is not None:
            random.seed(seed_order)
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        current_set = []
        for x in candidates:
            if time.time() > deadline:
                break
            if _is_valid_mean_free_with_dp(current_set, x):
                current_set.append(x)
        return current_set

    # Run a few greedy restarts
    for i in range(20):
        if time.time() > deadline:
            break
        S = greedy_generate(42 + i)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S[:]
        
        # Local Search
        ls_start = time.time()
        ls_deadline = ls_start + 4.0 # Tighter budget per restart to allow more restarts
        no_improvement_count = 0
        
        while time.time() < ls_deadline:
            improved = False
            
            # Try to add
            for _ in range(10):
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
                if len(S) > best_len:
                    best_len = len(S)
                    best_set = S[:]
                continue
            
            # Try swap
            for _ in range(10):
                if time.time() > ls_deadline:
                    break
                a = random.choice(S)
                x = random.randint(1, 512)
                if x in S:
                    continue
                others = [v for v in S if v != a]
                if _is_valid_mean_free_with_dp(others, x):
                    S.remove(a)
                    S.append(x)
                    improved = True
                    no_improvement_count = 0
                    break
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count > 20:
                    break

        if best_len >= 48:
            break

    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
