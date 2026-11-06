"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a Sidon set containing 0 with all differences 1..30 present,
    minimizing the largest element.
    
    Approach: Greedy set-packing/coverage using bitsets for difference tracking.
    We maintain a set of chosen elements and a bitset of used differences.
    At each step, we try to add an element that covers new required differences
    while keeping the Sidon property (all positive differences distinct).
    """
    import itertools
    
    REQUIRED = set(range(1, 31))
    
    # We'll use a greedy approach:
    # Start with {0}. Repeatedly add an element x (from a candidate pool)
    # that maximizes the number of new required differences covered,
    # subject to maintaining the Sidon property.
    # 
    # Key insight: For a Sidon set, all pairwise differences must be distinct.
    # We track used differences in a set. When adding x to set S, the new
    # differences are {x - s for s in S if x > s} and {s - x for s in S if s > x}.
    # All these must be new (not in used_diffs) and we check they don't collide.
    
    # Candidate pool: try small integers first to minimize diameter
    # We'll search for elements up to some limit
    
    def is_valid_addition(S, x, used_diffs):
        """Check if adding x to S maintains Sidon property."""
        new_diffs = []
        for s in S:
            if x > s:
                d = x - s
                if d in used_diffs:
                    return False, set()
                new_diffs.append(d)
            elif s > x:
                d = s - x
                if d in used_diffs:
                    return False, set()
                new_diffs.append(d)
        
        # Also check that new_diffs themselves are all distinct
        if len(set(new_diffs)) != len(new_diffs):
            return False, set()
        
        return True, set(new_diffs)
    
    def greedy_construct():
        S = [0]
        used_diffs = set()
        covered = set()
        
        # Candidate pool: try integers from 1 upward
        # We'll iterate through candidates and pick the best one each round
        max_iter = 100
        for _ in range(max_iter):
            if covered >= REQUIRED:
                break
            
            best_x = None
            best_new_covered = 0
            best_new_diffs = set()
            
            # Try candidates from 1 up to current max + some margin
            # To keep diameter small, prefer small candidates
            max_candidate = max(S) + 35 if S else 35
            
            for x in range(1, max_candidate + 1):
                if x in S:
                    continue
                
                valid, new_diffs = is_valid_addition(S, x, used_diffs)
                if not valid:
                    continue
                
                # Count how many new required differences this covers
                new_covered = new_diffs & (REQUIRED - covered)
                if len(new_covered) > best_new_covered:
                    best_new_covered = len(new_covered)
                    best_x = x
                    best_new_diffs = new_diffs
            
            if best_x is None:
                # No valid candidate found in range; expand search
                # Try larger candidates
                found = False
                for x in range(max_candidate + 1, max_candidate + 200):
                    if x in S:
                        continue
                    valid, new_diffs = is_valid_addition(S, x, used_diffs)
                    if not valid:
                        continue
                    new_covered = new_diffs & (REQUIRED - covered)
                    if len(new_covered) > best_new_covered:
                        best_new_covered = len(new_covered)
                        best_x = x
                        best_new_diffs = new_diffs
                        found = True
                if not found:
                    # Fallback: just add a large element to avoid getting stuck
                    x = max(S) + 100
                    while x in S:
                        x += 1
                    valid, new_diffs = is_valid_addition(S, x, used_diffs)
                    if not valid:
                        x += 1
                        valid, new_diffs = is_valid_addition(S, x, used_diffs)
                    best_x = x
                    best_new_diffs = new_diffs
            
            S.append(best_x)
            used_diffs |= best_new_diffs
            covered |= (best_new_diffs & REQUIRED)
        
        return S
    
    S = greedy_construct()
    
    # Verify
    S = sorted(set(S))
    diffs = set()
    valid = True
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d = S[j] - S[i]
            if d in diffs:
                valid = False
                break
            diffs.add(d)
        if not valid:
            break
    
    if not valid:
        # Fallback to a known construction
        # Use a simple Sidon set construction
        S = [0]
        used_diffs = set()
        for d in range(1, 31):
            # Try to add element that creates difference d
            for x in range(1, 200):
                if x in S:
                    continue
                new_diffs = set()
                ok = True
                for s in S:
                    diff = abs(x - s)
                    if diff in used_diffs or diff in new_diffs:
                        ok = False
                        break
                    new_diffs.add(diff)
                if ok and d in new_diffs:
                    S.append(x)
                    used_diffs |= new_diffs
                    break
            else:
                # Add element that creates d with 0
                if d not in S:
                    S.append(d)
                    for s in S[:-1]:
                        used_diffs.add(abs(d - s))
    
    return sorted(set(S))
# EVOLVE-BLOCK-END
