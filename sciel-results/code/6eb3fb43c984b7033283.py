import time
import random
import sys

# Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
def precompute_triples(max_n=180):
    triples = []
    for a in range(1, max_n + 1):
        for b in range(a + 1, max_n + 1):
            # 1/a = 1/b + 1/c => 1/c = 1/a - 1/b = (b-a)/(ab) => c = ab/(b-a)
            # c must be integer and c > b
            if (b - a) == 0:
                continue
            if (a * b) % (b - a) == 0:
                c = (a * b) // (b - a)
                if c > b and c <= max_n:
                    triples.append((a, b, c))
    return triples

TRIPLES = precompute_triples(180)
NUM_TRIPLES = len(TRIPLES)

def is_valid(S):
    S_set = set(S)
    for a, b, c in TRIPLES:
        if a in S_set and b in S_set and c in S_set:
            return False
    return True

def get_conflicts(S_set, x):
    """
    Returns a list of triples that contain x and at least 2 other elements in S_set.
    Actually, we just need to know if adding x creates a conflict.
    A conflict is a triple (a,b,c) where x is one of them, and the other two are in S_set.
    """
    conflicts = []
    for a, b, c in TRIPLES:
        if x == a:
            if b in S_set and c in S_set:
                conflicts.append((a, b, c))
        elif x == b:
            if a in S_set and c in S_set:
                conflicts.append((a, b, c))
        elif x == c:
            if a in S_set and b in S_set:
                conflicts.append((a, b, c))
    return conflicts

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    best_set = set(range(91, 181))  # Initial seed
    best_size = len(best_set)
    
    # Ensure initial set is valid
    if not is_valid(best_set):
        # Fallback to greedy if seed is invalid (it should be valid)
        S = set()
        for i in range(180, 0, -1):
            S.add(i)
            if not is_valid(S):
                S.remove(i)
        best_set = S
        best_size = len(S)

    all_elements = list(range(1, 181))
    
    while time.time() < deadline:
        # Randomized Greedy Initialization
        random.shuffle(all_elements)
        S = set()
        for x in all_elements:
            S.add(x)
            if not is_valid(S):
                S.remove(x)
        
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)
            
        # Local Search
        S_list = list(S)
        current_size = len(S)
        
        # Try to improve S
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding missing elements
            missing = [x for x in all_elements if x not in S]
            random.shuffle(missing)
            
            for x in missing:
                if time.time() >= deadline:
                    break
                
                # Check if adding x is valid
                S.add(x)
                if is_valid(S):
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_set = S.copy()
                        best_size = current_size
                else:
                    S.remove(x)
                    # Try swap: remove one conflicting element, add x
                    # Find conflicts
                    S_set = S
                    # We know it's invalid, so there is at least one conflict
                    # Find all triples involving x and elements in S
                    # Instead of full re-check, we can just pick a random element from S to remove
                    # and see if it fixes the conflict.
                    # But we need to be sure.
                    
                    # Efficient check:
                    # If adding x causes conflict, there is a triple (a,b,c) with x in it and other two in S.
                    # Let's find one such triple.
                    conflict_found = False
                    for a, b, c in TRIPLES:
                        if x == a:
                            if b in S_set and c in S_set:
                                # Remove b or c
                                if random.random() < 0.5:
                                    S.remove(b)
                                else:
                                    S.remove(c)
                                conflict_found = True
                                break
                        elif x == b:
                            if a in S_set and c in S_set:
                                if random.random() < 0.5:
                                    S.remove(a)
                                else:
                                    S.remove(c)
                                conflict_found = True
                                break
                        elif x == c:
                            if a in S_set and b in S_set:
                                if random.random() < 0.5:
                                    S.remove(a)
                                else:
                                    S.remove(b)
                                conflict_found = True
                                break
                    
                    if conflict_found:
                        # Now S has x added and one element removed.
                        # Size is same as before.
                        # Check if valid now?
                        # Removing one element from a conflicting triple should resolve that specific conflict.
                        # But might create others? No, removing elements only helps validity.
                        # So S is now valid.
                        # But size didn't increase.
                        # We mark improved to continue search?
                        # Actually, this move doesn't increase size.
                        # We only mark improved if size increases.
                        # So we don't set improved = True here.
                        # But we changed S.
                        # Let's just continue.
                        pass

        # If we found a better set, update best
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)

    return list(best_set)
