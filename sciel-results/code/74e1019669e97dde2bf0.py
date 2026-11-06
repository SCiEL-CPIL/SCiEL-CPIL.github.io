import time
import bisect

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 256
    
    # Precompute: for each a, list of (b,c) pairs where b>c>a and a|(b+c)
    # This lets us check validity of adding a by checking if any such pair is in S
    forbidden_pairs = {}
    for a in range(1, n + 1):
        pairs = []
        for b in range(a + 1, n + 1):
            for c in range(b, n + 1):
                if (b + c) % a == 0:
                    pairs.append((b, c))
        forbidden_pairs[a] = pairs
    
    def is_valid_addition(a, S_set):
        """Check if a can be added to S. All elements in S are > a."""
        for (b, c) in forbidden_pairs[a]:
            if b in S_set and c in S_set:
                return False
        return True
    
    def full_validate(S_list):
        n2 = len(S_list)
        S_set = set(S_list)
        for i in range(n2):
            a = S_list[i]
            for (b, c) in forbidden_pairs[a]:
                if b in S_set and c in S_set:
                    return False
        return True
    
    best_S = set(range(171, n + 1))
    best_size = len(best_S)
    
    # Try starting from various thresholds, descending
    for m_start in [165, 160, 155, 150, 145, 140, 135, 130, 128, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]:
        if time.time() > deadline - 5:
            break
        
        S_list = list(range(m_start, n + 1))
        S_set = set(S_list)
        
        # Check if base interval [m_start, n] is valid
        valid = True
        for i in range(len(S_list)):
            a = S_list[i]
            for (b, c) in forbidden_pairs[a]:
                if b in S_set and c in S_set:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        if len(S_list) > best_size:
            best_S = set(S_list)
            best_size = len(S_list)
        
        # Add elements below m_start, descending
        for a in range(m_start - 1, 0, -1):
            if time.time() > deadline - 2:
                break
            if a in S_set:
                continue
            if is_valid_addition(a, S_set):
                pos = bisect.bisect_left(S_list, a)
                S_list.insert(pos, a)
                S_set.add(a)
                if len(S_list) > best_size:
                    best_S = set(S_list)
                    best_size = len(S_list)
    
    # Try to add any remaining elements in multiple passes
    if time.time() < deadline - 10:
        S_list = sorted(best_S)
        S_set = set(S_list)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Try elements in ascending order to find small elements that fit
            for a in range(1, n + 1):
                if a in S_set:
                    continue
                if is_valid_addition(a, S_set):
                    pos = bisect.bisect_left(S_list, a)
                    S_list.insert(pos, a)
                    S_set.add(a)
                    changed = True
                    if len(S_list) > best_size:
                        best_S = set(S_list)
                        best_size = len(S_list)
    
    # Final validation
    if time.time() < deadline:
        S_final = sorted(best_S)
        if len(S_final) > 5 and not full_validate(S_final):
            S_final = list(range(171, n + 1))
        return S_final
    
    return sorted(best_S)
