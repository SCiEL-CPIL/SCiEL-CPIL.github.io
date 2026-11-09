"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_valid_triple(a, b, c, S):
    """Check if adding a,b,c creates a forbidden configuration.
    Actually, we check if a,b,c are in S and all three pair sums are in S."""
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s
    
    n = 128
    # Start with the known good set {64, ..., 128}
    S = set(range(64, n + 1))
    
    # Try to add elements from 1 to 63
    # For each candidate x, check if adding x to S creates any forbidden triple
    # A forbidden triple: distinct a,b,c in S where a+b, a+c, b+c all in S
    
    # Efficient check: when adding x, new forbidden triples must include x
    # So check triples (x, a, b) for a,b in S, a != b, x != a, x != b
    # where x+a, x+b, a+b are all in S ∪ {x}
    
    # But since x is new, the sums x+a and x+b are new sums
    # a+b must already be in S (since a,b were already in S)
    # So we need: a+b in S, and x+a in S, and x+b in S
    # But x+a and x+b are > x, and since x <= 63 and a,b >= 64 (initially),
    # x+a could be in S if x+a <= 128
    
    # Let's do a greedy: try to add each x from 1 to 63, in some order
    # Order: try smaller x first? Or larger? Let's try larger x first (closer to 64)
    
    # Actually, let's be more careful. Let's try adding elements one by one
    # and check validity.
    
    def check_validity(S_set):
        """Check if S_set has any forbidden triple."""
        elems = sorted(S_set)
        # For efficiency, check all triples
        # But this is O(k^3) which for k~65 is ~274k, fine
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                for k in range(j+1, len(elems)):
                    a, b, c = elems[i], elems[j], elems[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        return False
        return True
    
    # Greedy: try to add elements from 63 down to 1
    for x in range(63, 0, -1):
        if time.time() > deadline:
            break
        S_new = S | {x}
        if check_validity(S_new):
            S = S_new
    
    # Now try to add more by removing some elements and adding others? 
    # For now, let's also try a different initial set and greedy
    
    # Convert to list of Python ints
    result = sorted(S)
    # Ensure they are Python ints
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
