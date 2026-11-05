import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0

    def is_sidon_compatible(x, A, used_sums):
        if 2 * x in used_sums:
            return False
        for y in A:
            if x + y in used_sums:
                return False
        return True

    def get_newly_blocked(x, A, used_sums, unblocked):
        new_used_sums = set(used_sums)
        new_used_sums.add(2 * x)
        for y in A:
            new_used_sums.add(x + y)

        newly_blocked = []
        for z in unblocked:
            if z == x:
                continue
            if 2 * z in new_used_sums:
                newly_blocked.append(z)
                continue
            blocked = False
            for y in A:
                if z + y in new_used_sums:
                    blocked = True
                    break
            if not blocked and z + x in new_used_sums:
                blocked = True
            if blocked:
                newly_blocked.append(z)
        return newly_blocked

    def greedy_max_coverage(init_A=None, randomize_ties=False, bias=0.0):
        if init_A is None:
            A = []
        else:
            A = list(init_A)
        
        used_sums = set()
        for x in A:
            used_sums.add(2 * x)
            for y in A:
                if x < y: # Avoid duplicates and self if handled? 
                    # Actually, just add all sums x+y for x,y in A
                    pass
            # Simpler: rebuild used_sums from scratch
        used_sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                used_sums.add(A[i] + A[j])

        blocked = set()
        unblocked = set(range(1, N + 1))
        for x in A:
            if x in unblocked:
                unblocked.remove(x)
                blocked.add(x)
        
        # Note: Elements in A are not "missing", so they don't need to be blocked.
        # But they might block others.
        # The unblocked set contains elements NOT in A.
        # We need to block all of them.

        while unblocked:
            best_x = None
            best_score = -1
            best_newly_blocked = []
            
            for x in range(1, N + 1):
                if x in A:
                    continue
                if not is_sidon_compatible(x, A, used_sums):
                    continue
                
                newly_blocked = get_newly_blocked(x, A, used_sums, unblocked)
                score = len(newly_blocked)
                
                if bias != 0.0:
                    # Bias towards smaller or larger x
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
            used_sums.add(2 * best_x)
            for y in A:
                used_sums.add(best_x + y)
            
            for z in best_newly_blocked:
                if z in unblocked:
                    unblocked.remove(z)
                    blocked.add(z)
            
            if best_x in unblocked:
                unblocked.remove(best_x)
                blocked.add(best_x)

        return A

    # Initial deterministic run
    best_set = greedy_max_coverage()
    best_size = len(best_set)
    
    # Randomized restarts
    while time.time() < deadline:
        # Try different biases and randomizations
        r = random.random()
        if r < 0.3:
            current_set = greedy_max_coverage(randomize_ties=True)
        elif r < 0.6:
            current_set = greedy_max_coverage(randomize_ties=True, bias=0.01)
        else:
            # Start with a random element
            init_x = random.randint(1, N)
            current_set = greedy_max_coverage(init_A=[init_x], randomize_ties=True)
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set

    return best_set
