import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    LO = 10000
    HI = 1000000
    
    # Primes to check
    # We start with small primes and go up
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    def get_smooth_numbers(primes, lo, hi):
        # Generate all numbers in [lo, hi] whose prime factors are in primes
        # Use a set to store unique numbers
        smooths = set()
        # Start with 1
        # Use a heap to generate in increasing order? Or just recursive?
        # Recursive might be deep but primes are small.
        # Or iterative multiplication.
        
        # Better: Use a min-heap to generate all p-smooth numbers up to hi
        # But we only care about those >= lo.
        
        # Simple approach:
        # Since hi is 1e6, and p is small, the number of smooth numbers is small.
        # We can just generate all divisors of (product of primes)^k? No.
        
        # Let's use a heap to generate all p-smooth numbers <= hi
        # Then filter for >= lo
        
        heap = [1]
        heapq.heapify(heap)
        visited = set([1])
        
        while heap:
            x = heapq.heappop(heap)
            if x > hi:
                break
            if x >= lo:
                smooths.add(x)
            
            for p in primes:
                nx = x * p
                if nx <= hi and nx not in visited:
                    visited.add(nx)
                    heapq.heappush(heap, nx)
                    
        return smooths

    best_n = LO # Fallback
    
    for p in primes:
        if time.time() > deadline:
            break
            
        smooths = get_smooth_numbers([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113][:primes.index(p)+1], LO, HI)
        
        # Check for consecutive pairs
        found = False
        # Sort smooths to iterate in order, or just check set membership
        # Checking set membership is O(1)
        for x in smooths:
            if (x + 1) in smooths:
                # Found consecutive pair x, x+1
                # x is in [LO, HI-1]
                # x+1 is in [LO+1, HI]
                # Both are p-smooth
                # This is the first p that works, so this is optimal
                best_n = x
                found = True
                break
        
        if found:
            return int(best_n)
            
    return int(best_n)
