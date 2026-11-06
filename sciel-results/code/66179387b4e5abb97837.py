import time
from itertools import combinations

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Modular primes for pre-filtering
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    def modular_violates(selected, candidate, primes_list):
        """Check if candidate violates mean condition modulo any prime.
        Returns True if violation detected (candidate should be rejected)."""
        others = selected  # all currently selected elements
        n = len(others)
        if n < 2:
            return False
        
        for p in primes_list:
            # For each possible subset size k >= 2, check if any subset of 'others'
            # with size k has sum ≡ k * candidate (mod p)
            # We use DP: dp[k][s] = True if some subset of size k has sum ≡ s (mod p)
            max_sum_mod = p  # sums are in 0..p-1
            # dp[k] is a set of achievable residues mod p
            dp = [set() for _ in range(n + 1)]
            dp[0].add(0)
            
            for x in others:
                xm = x % p
                for k in range(len(dp) - 1, 0, -1):
                    for s in list(dp[k - 1]):
                        dp[k].add((s + xm) % p)
            
            # Check if any k >= 2 has k * candidate % p in dp[k]
            for k in range(2, n + 1):
                target = (k * candidate) % p
                if target in dp[k]:
                    return True
        
        return False
    
    def exact_violates(selected, candidate):
        """Exact check: does any subset of selected (size >= 2) have mean = candidate?"""
        others = selected
        n = len(others)
        if n < 2:
            return False
        
        # For each k >= 2, check if any subset of size k sums to k * candidate
        # Use bitset DP for subset sums
        target_max = sum(others)  # max possible sum
        # We need: for some k>=2, subset sum = k * candidate
        # Bitset: bit i set means sum i is achievable
        # But we also need to track size... 
        # Alternative: for each k, check if k*candidate is achievable by subset of size k
        
        # Since n is at most ~48, and we check per k, use DP with (size, sum)
        # But sum can be up to 48*512 = 24576, which is manageable
        max_sum = sum(others)
        if max_sum == 0:
            return False
        
        # dp[k] = set of achievable sums using exactly k elements
        # But this could be expensive. Let's use bitset per k.
        # Actually, for n <= 48, the number of distinct sums is bounded.
        # Let's use a different approach: for each k from 2 to n,
        # check if k*candidate can be formed by k elements.
        
        # Use DP: reachable[k][s] but that's O(n * max_sum) per k... 
        # Better: single DP tracking (count, sum) pairs
        # dp is a set of (count, sum) pairs, but that could explode.
        
        # Alternative: for each k, use bitset where bit position represents sum
        # dp_k is a bitset of achievable sums using exactly k elements
        # Start: dp_0 has bit 0 set
        # For each element x, update dp_k |= (dp_{k-1} << x)
        
        # max_sum is at most 48*512 = 24576
        # Use Python integers as bitsets
        
        dp = [0] * (n + 1)
        dp[0] = 1  # bit 0 set: sum 0 achievable with 0 elements
        
        for x in others:
            for k in range(n, 0, -1):
                dp[k] |= (dp[k-1] << x)
        
        for k in range(2, n + 1):
            target = k * candidate
            if target <= max_sum and (dp[k] >> target) & 1:
                return True
        
        return False
    
    selected = []
    
    # Try candidates in various orders
    # First, try a greedy approach with modular pre-filter
    
    # Order: try to maximize count. Start with spread-out numbers.
    # Strategy: iterate through 1..512, use modular filter first, then exact check
    
    # For better results, try multiple orderings
    best = []
    
    # Ordering 1: natural order
    for order_seed in range(0, 20):
        if time.time() > deadline - 5:
            break
        
        if order_seed == 0:
            candidates = list(range(1, 513))
        elif order_seed == 1:
            candidates = list(range(512, 0, -1))
        else:
            import random
            rng = random.Random(order_seed)
            candidates = list(range(1, 513))
            rng.shuffle(candidates)
        
        selected = []
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if len(selected) >= 48:
                break
            
            # Modular pre-filter
            if modular_violates(selected, x, primes):
                continue
            
            # Exact check
            if exact_violates(selected, x):
                continue
            
            selected.append(x)
        
        if len(selected) > len(best):
            best = selected[:]
            if len(best) >= 48:
                break
    
    return best
