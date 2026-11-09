import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def is_powerful(n):
        """Check if n is powerful using trial division."""
        if n < 2:
            return True
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                # Count exponent
                exp = 0
                while temp % d == 0:
                    temp //= d
                    exp += 1
                if exp < 2:
                    return False
            d += 1
        # Check remaining factor
        if temp > 1:
            # temp is prime, exponent is 1
            return False
        return True

    def generate_powerful_up_to(limit, max_count=100000):
        """Generate powerful numbers up to limit using the n^2 * k^3 form."""
        powerful = set()
        max_k = int(limit ** (1/3)) + 2
        max_n = int(math.isqrt(limit)) + 2
        
        for k in range(1, max_k + 1):
            k3 = k ** 3
            if k3 > limit:
                break
            max_n_for_k = int(math.isqrt(limit // k3))
            for n in range(1, max_n_for_k + 1):
                x = n * n * k3
                if x <= limit:
                    powerful.add(x)
                if len(powerful) > max_count:
                    break
            if len(powerful) > max_count:
                break
        
        return sorted(powerful)

    def count_representations(N, powerful_set, powerful_list):
        """Count the number of pairs (a,b) with a<=b, a+b=N, a,b powerful."""
        count = 0
        pairs = []
        for a in powerful_list:
            if a > N:
                break
            b = N - a
            if b < a:
                continue
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        return count, pairs

    def estimate_expected_reps(N, powerful_list, sample_size=1000):
        """Estimate expected number of representations using sampling."""
        if not powerful_list:
            return 0
        
        # Sample from powerful numbers
        if len(powerful_list) <= sample_size:
            sample = powerful_list
        else:
            sample = np.random.choice(powerful_list, size=sample_size, replace=False)
        
        count = 0
        for a in sample:
            if a > N:
                continue
            b = N - a
            if b >= a and b in powerful_set:
                count += 1
        
        # Scale up
        return count * len(powerful_list) / len(sample)

    # Phase 1: Generate powerful numbers up to a moderate limit
    # We'll use multiple scales to balance coverage
    powerful_cache = {}
    
    def get_powerful(limit):
        if limit in powerful_cache:
            return powerful_cache[limit]
        P = generate_powerful_up_to(limit)
        powerful_cache[limit] = P
        return P

    # Start with moderate limit to get a good set
    M1 = 50_000_000
    P1 = get_powerful(M1)
    P1_set = set(P1)
    
    # Phase 2: Use probabilistic heuristic to find candidate N values
    # Strategy: For various N, estimate E[reps(N)] and pick the best
    # We'll search in ranges where N is a sum of two powerful numbers from our set
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1
    
    # Try candidate N values based on sums of powerful numbers
    # Sample some sums to find promising regions
    np.random.seed(42)
    
    # Generate candidate N values by summing random pairs of powerful numbers
    # But focus on N where both summands are "typical" powerful numbers
    if len(P1) >= 100:
        # Sample pairs to get candidate N values
        num_samples = min(5000, len(P1) * len(P1) // 10)
        candidate_Ns = set()
        
        for _ in range(num_samples):
            i = np.random.randint(0, len(P1))
            j = np.random.randint(0, len(P1))
            a = P1[i]
            b = P1[j]
            N = a + b
            if N <= 10**18:
                candidate_Ns.add(N)
        
        # Also add some structured candidates: N = 2*k for k in P1 (diagonal)
        for a in P1[:1000]:
            candidate_Ns.add(2 * a)
        
        # Sort candidates by some heuristic - prefer N where many powerful numbers
        # in our set are "close" to N/2 (more likely to have complementary powerful)
        candidate_Ns = list(candidate_Ns)
        
        # Estimate and verify top candidates
        # We'll verify up to 200 candidates
        verified = 0
        max_verify = 200
        
        # Sort by a quick heuristic: N/2 should be in a "dense" region
        # Simple heuristic: prefer N where N/2 is close to a powerful number
        def heuristic_score(N):
            half = N // 2
            # Find closest powerful number to half
            idx = np.searchsorted(P1, half)
            if idx < len(P1):
                return abs(P1[idx] - half)
            if idx > 0:
                return abs(P1[idx-1] - half)
            return 10**18
        
        candidate_Ns.sort(key=heuristic_score)
        
        for N in candidate_Ns[:max_verify]:
            if time.time() > deadline - 10:
                break
            
            count, pairs = count_representations(N, P1_set, P1)
            
            if count > best_count:
                best_count = count
                best_result = {'N': N, 'pairs': pairs}
            
            verified += 1
    
    # Phase 3: If we have time, try to extend to larger N by generating more powerful numbers
    if time.time() < deadline - 30 and best_count < 10:
        # Try a larger limit
        M2 = 200_000_000
        P2 = get_powerful(M2)
        P2_set = set(P2)
        
        # Re-verify best N with larger set
        count2, pairs2 = count_representations(best_result['N'], P2_set, P2)
        if count2 > best_count:
            best_count = count2
            best_result = {'N': best_result['N'], 'pairs': pairs2}
        
        # Also try new candidates from the larger set
        if len(P2) >= 100:
            num_samples2 = min(3000, len(P2) * len(P2) // 20)
            candidate_Ns2 = set()
            
            for _ in range(num_samples2):
                i = np.random.randint(0, len(P2))
                j = np.random.randint(0, len(P2))
                a = P2[i]
                b = P2[j]
                N = a + b
                if N <= 10**18:
                    candidate_Ns2.add(N)
            
            candidate_Ns2 = list(candidate_Ns2)
            candidate_Ns2.sort(key=heuristic_score)
            
            for N in candidate_Ns2[:100]:
                if time.time() > deadline - 10:
                    break
                
                count, pairs = count_representations(N, P2_set, P2)
                
                if count > best_count:
                    best_count = count
                    best_result = {'N': N, 'pairs': pairs}
    
    return best_result
