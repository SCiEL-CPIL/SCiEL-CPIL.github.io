import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # safety margin before 30s limit

    def is_sidon(A):
        """Check if A is a Sidon set (all unordered sums a+b with a<=b are distinct)."""
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(A):
        """Check if A is inclusion-maximal: adding any element from [1,N]\A destroys Sidon property."""
        A_set = set(A)
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        
        for x in range(1, N + 1):
            if x in A_set:
                continue
            # Check if adding x would violate Sidon property
            compatible = True
            if 2 * x in sums:
                compatible = False
            else:
                for y in A:
                    if x + y in sums:
                        compatible = False
                        break
            if compatible:
                return False  # Can add x, so not maximal
        return True

    def greedy_max_coverage(init_A=None, randomize_ties=False, bias=0.0, seed=None):
        """Greedy: at each step pick the element that blocks the most remaining unblocked elements."""
        if seed is not None:
            random.seed(seed)
        
        A = list(init_A) if init_A else []
        A_set = set(A)
        
        # Build used_sums
        used_sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                used_sums.add(A[i] + A[j])
        
        # unblocked: elements not in A that are still potentially addable
        unblocked = set(range(1, N + 1)) - A_set
        
        while unblocked:
            best_x = None
            best_score = -1
            best_newly_blocked = []
            
            for x in range(1, N + 1):
                if x in A_set:
                    continue
                # Check Sidon compatibility
                if 2 * x in used_sums:
                    continue
                compatible = True
                for y in A:
                    if x + y in used_sums:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Count how many currently-unblocked elements would be blocked
                new_sums = set(used_sums)
                new_sums.add(2 * x)
                for y in A:
                    new_sums.add(x + y)
                
                newly_blocked = 0
                for z in unblocked:
                    if z == x:
                        continue
                    if 2 * z in new_sums:
                        newly_blocked += 1
                        continue
                    blocked = False
                    for y in A:
                        if z + y in new_sums:
                            blocked = True
                            break
                    if not blocked and z + x in new_sums:
                        blocked = True
                    if blocked:
                        newly_blocked += 1
                
                score = newly_blocked
                if bias != 0.0:
                    if bias > 0:
                        score -= x * bias
                    else:
                        score += x * abs(bias)
                if randomize_ties:
                    score += random.uniform(0, 0.5)
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_newly_blocked = newly_blocked
            
            if best_x is None:
                break
            
            A.append(best_x)
            A_set.add(best_x)
            used_sums.add(2 * best_x)
            for y in A:
                if y != best_x:
                    used_sums.add(best_x + y)
            
            unblocked.discard(best_x)
        
        return A

    def local_search_improve(A):
        """Try to reduce size via swap/remove operations while maintaining maximality."""
        A = list(A)
        improved = True
        while improved:
            improved = False
            # Try removing each element and re-saturating
            for i in range(len(A)):
                candidate = A[:i] + A[i+1:]
                if len(candidate) == 0:
                    continue
                # Check if candidate is Sidon
                if not is_sidon(candidate):
                    continue
                # Greedily saturate
                saturated = greedy_max_coverage(init_A=candidate)
                if len(saturated) < len(A) and is_maximal(saturated):
                    A = saturated
                    improved = True
                    break
        return A

    # Initial deterministic run
    best_set = greedy_max_coverage()
    best_size = len(best_set)
    
    # Try local search on the initial result
    improved = local_search_improve(best_set)
    if len(improved) < best_size:
        best_size = len(improved)
        best_set = improved

    # Randomized restarts with various strategies
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        r = random.random()
        
        if r < 0.25:
            # Pure randomized greedy
            current_set = greedy_max_coverage(randomize_ties=True, seed=attempt)
        elif r < 0.45:
            # Greedy with small bias toward smaller numbers
            current_set = greedy_max_coverage(randomize_ties=True, bias=0.005, seed=attempt)
        elif r < 0.65:
            # Greedy with bias toward larger numbers
            current_set = greedy_max_coverage(randomize_ties=True, bias=-0.005, seed=attempt)
        elif r < 0.85:
            # Start with a random seed element
            init_x = random.randint(1, N)
            current_set = greedy_max_coverage(init_A=[init_x], randomize_ties=True, seed=attempt)
        else:
            # Start with two random compatible elements
            x1 = random.randint(1, N)
            x2 = random.randint(1, N)
            while x2 == x1 or x1 + x2 == 2*x1 or x1 + x2 == 2*x2:
                x2 = random.randint(1, N)
            init = [x1, x2]
            if is_sidon(init):
                current_set = greedy_max_coverage(init_A=init, randomize_ties=True, seed=attempt)
            else:
                current_set = greedy_max_coverage(randomize_ties=True, seed=attempt)
        
        # Apply local search
        improved = local_search_improve(current_set)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved
    
    return best_set
