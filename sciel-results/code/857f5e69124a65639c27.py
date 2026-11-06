"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill
    
    def build_sum_set(S):
        """Build the set of all unordered pair sums a+b with a<=b."""
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                sums.add(a + b)
        return sums
    
    def is_valid_sidon(S):
        """Check if S is a valid Sidon set."""
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def can_add(S, sums, x):
        """Check if x can be added to S (with its sums) maintaining Sidon property."""
        for a in S:
            s = x + a
            if s in sums:
                return False
        s = 2 * x
        if s in sums:
            return False
        return True
    
    def is_maximal(S, sums):
        """Check if S is maximal: no missing element can be added."""
        S_set = set(S)
        for x in range(1, N + 1):
            if x not in S_set:
                if can_add(S, sums, x):
                    return False
        return True
    
    def greedy_build_maximal():
        """Build a maximal Sidon set by greedy addition."""
        S = []
        sums = set()
        remaining = list(range(1, N + 1))
        
        while remaining:
            if time.time() > deadline:
                break
            # Pick element that blocks most others
            best_x = None
            best_score = -1
            for x in remaining:
                if not can_add(S, sums, x):
                    continue
                # Count how many remaining elements would be blocked
                new_sums = sums | {x + a for a in S} | {2 * x}
                blocked = 0
                for y in remaining:
                    if y == x:
                        continue
                    if not can_add(S + [x], new_sums, y):
                        blocked += 1
                score = blocked
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            S.append(best_x)
            for a in S:
                sums.add(best_x + a)
            sums.add(2 * best_x)
            remaining.remove(best_x)
        
        return S, sums
    
    def try_remove_element(S, sums, elem):
        """Try to remove elem from S. Return (new_S, new_sums, success) if removal keeps maximality."""
        new_S = [x for x in S if x != elem]
        if not new_S:
            return None, None, False
        
        new_sums = build_sum_set(new_S)
        
        # Check if still maximal
        if not is_maximal(new_S, new_sums):
            return None, None, False
        
        return new_S, new_sums, True
    
    def minimize_by_removal(S, sums):
        """Iteratively try to remove elements to minimize size while keeping maximality."""
        S = list(S)
        sums = build_sum_set(S)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                elem = S[i]
                new_S, new_sums, success = try_remove_element(S, sums, elem)
                if success:
                    S = new_S
                    sums = new_sums
                    improved = True
                    break  # Restart loop after successful removal
        
        return S, sums
    
    # Main algorithm
    best_S = None
    best_size = N + 1
    
    # Try multiple random orderings for greedy construction
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Random permutation for diversity
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        
        # Greedy build with random tie-breaking
        S = []
        sums = set()
        
        for x in elements:
            if time.time() > deadline:
                break
            if can_add(S, sums, x):
                S.append(x)
                for a in S:
                    sums.add(x + a)
                sums.add(2 * x)
        
        # Check if maximal
        if not is_maximal(S, sums):
            continue
        
        # Minimize by removal
        S_min, sums_min = minimize_by_removal(S, sums)
        
        if len(S_min) < best_size:
            best_size = len(S_min)
            best_S = S_min
            sums = sums_min
    
    # Fallback: if no valid set found, use simple greedy
    if best_S is None:
        S, sums = greedy_build_maximal()
        if S:
            best_S, sums = minimize_by_removal(S, sums)
        else:
            best_S = [1]
    
    return best_S
# EVOLVE-BLOCK-END
