import random
import time

def construct():
    random.seed(42)
    A = list(range(1, 21))
    max_val = 10000
    
    def compute_scores(A):
        sumset = set()
        prodset = set()
        for i in range(20):
            for j in range(i, 20):
                sumset.add(A[i] + A[j])
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
    
    current_score = compute_scores(A)
    start_time = time.time()
    timeout = 55
    
    # Local search: try replacing elements
    while time.time() - start_time < timeout:
        improved = False
        
        # Strategy 1: Replace with random values
        for i in range(20):
            if time.time() - start_time > timeout:
                break
            old_val = A[i]
            candidates = random.sample(range(1, max_val + 1), 300)
            for new_val in candidates:
                if new_val in A:
                    continue
                A[i] = new_val
                A_sorted = sorted(A)
                new_score = compute_scores(A_sorted)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    A = A_sorted
                    break
                else:
                    A[i] = old_val
            if improved:
                break
        
        if improved:
            continue
        
        # Strategy 2: Replace with values near existing elements
        improved2 = False
        for i in range(20):
            if time.time() - start_time > timeout:
                break
            old_val = A[i]
            for delta in range(-100, 101):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 1 or new_val > max_val or new_val in A:
                    continue
                A[i] = new_val
                A_sorted = sorted(A)
                new_score = compute_scores(A_sorted)
                if new_score < current_score:
                    current_score = new_score
                    improved2 = True
                    A = A_sorted
                    break
                else:
                    A[i] = old_val
            if improved2:
                break
        
        if not improved2:
            # Strategy 3: Random restart with a different structure
            # Try geometric-like sets
            for base in [2, 3, 4, 5]:
                if time.time() - start_time > timeout:
                    break
                geo = []
                val = 1
                while val <= max_val and len(geo) < 20:
                    geo.append(val)
                    val *= base
                if len(geo) >= 20:
                    geo = geo[:20]
                else:
                    # Fill with random values
                    while len(geo) < 20:
                        v = random.randint(1, max_val)
                        if v not in geo:
                            geo.append(v)
                geo = sorted(geo)
                geo_score = compute_scores(geo)
                if geo_score < current_score:
                    A = geo
                    current_score = geo_score
                    improved2 = True
                    break
        
        if not improved2:
            break
    
    return A
