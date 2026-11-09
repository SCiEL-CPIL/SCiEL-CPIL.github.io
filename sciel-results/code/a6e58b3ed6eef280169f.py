import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def is_powerful(n):
        """Check if n is powerful using trial division."""
        if n < 2:
            return True
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                exp = 0
                while temp % d == 0:
                    temp //= d
                    exp += 1
                if exp < 2:
                    return False
            d += 1
        if temp > 1:
            return False
        return True

    def generate_powerful_up_to(limit, max_count=500000):
        """Generate powerful numbers up to limit using the n^2 * k^3 form."""
        powerful = set()
        max_k = int(round(limit ** (1/3))) + 2
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

    def estimate_expected_reps(N, powerful_list, powerful_set):
        """Estimate expected number of representations using the full list."""
        count = 0
        for a in powerful_list:
            if a > N:
                break
            b = N - a
            if b >= a and b in powerful_set:
                count += 1
        return count

    # Phase 1: Generate powerful numbers up to a moderate limit
    M1 = 10_000_000
    P1 = generate_powerful_up_to(M1)
    P1_set = set(P1)
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1

    np.random.seed(42)
    
    # Generate candidate N values by summing random pairs of powerful numbers
    candidate_Ns = set()
    
    # Add structured candidates: N = 2*a for a in P1 (diagonal)
    for a in P1:
        N = 2 * a
        if N <= 10**18:
            candidate_Ns.add(N)
    
    # Sample random pairs
    num_samples = min(10000, len(P1) * 2)
    for _ in range(num_samples):
        i = np.random.randint(0, len(P1))
        j = np.random.randint(0, len(P1))
        a = P1[i]
        b = P1[j]
        N = a + b
        if N <= 10**18:
            candidate_Ns.add(N)
    
    candidate_Ns = list(candidate_Ns)
    
    # Sort by a heuristic: prefer N where N/2 is close to a powerful number
    def heuristic_score(N):
        half = N // 2
        idx = np.searchsorted(P1, half)
        if idx < len(P1):
            return abs(P1[idx] - half)
        if idx > 0:
            return abs(P1[idx-1] - half)
        return 10**18
    
    candidate_Ns.sort(key=heuristic_score)
    
    # Verify top candidates
    for N in candidate_Ns[:300]:
        if time_left() < 15:
            break
        
        count, pairs = count_representations(N, P1_set, P1)
        
        if count > best_count:
            best_count = count
            best_result = {'N': N, 'pairs': pairs}

    # Phase 2: If we have time, try larger N by generating more powerful numbers
    if time_left() > 40 and best_count < 15:
        M2 = 50_000_000
        P2 = generate_powerful_up_to(M2)
        P2_set = set(P2)
        
        # Re-verify best N with larger set
        count2, pairs2 = count_representations(best_result['N'], P2_set, P2)
        if count2 > best_count:
            best_count = count2
            best_result = {'N': best_result['N'], 'pairs': pairs2}
        
        # Generate new candidates from the larger set
        candidate_Ns2 = set()
        
        for a in P2:
            N = 2 * a
            if N <= 10**18:
                candidate_Ns2.add(N)
        
        num_samples2 = min(5000, len(P2))
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
        
        for N in candidate_Ns2[:150]:
            if time_left() < 15:
                break
            
            count, pairs = count_representations(N, P2_set, P2)
            
            if count > best_count:
                best_count = count
                best_result = {'N': N, 'pairs': pairs}

    # Phase 3: If we have time, try even larger N
    if time_left() > 60 and best_count < 25:
        M3 = 200_000_000
        P3 = generate_powerful_up_to(M3)
        P3_set = set(P3)
        
        # Re-verify best N with larger set
        count3, pairs3 = count_representations(best_result['N'], P3_set, P3)
        if count3 > best_count:
            best_count = count3
            best_result = {'N': best_result['N'], 'pairs': pairs3}
        
        # Generate new candidates
        candidate_Ns3 = set()
        
        for a in P3:
            N = 2 * a
            if N <= 10**18:
                candidate_Ns3.add(N)
        
        num_samples3 = min(3000, len(P3))
        for _ in range(num_samples3):
            i = np.random.randint(0, len(P3))
            j = np.random.randint(0, len(P3))
            a = P3[i]
            b = P3[j]
            N = a + b
            if N <= 10**18:
                candidate_Ns3.add(N)
        
        candidate_Ns3 = list(candidate_Ns3)
        candidate_Ns3.sort(key=heuristic_score)
        
        for N in candidate_Ns3[:100]:
            if time_left() < 15:
                break
            
            count, pairs = count_representations(N, P3_set, P3)
            
            if count > best_count:
                best_count = count
                best_result = {'N': N, 'pairs': pairs}

    return best_result
