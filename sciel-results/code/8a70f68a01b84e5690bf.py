import time
import random

def construct():
    start = time.time()
    deadline = start + 55.0
    N = 192

    def is_valid_set(S):
        """Verify set S is valid."""
        S_list = sorted(S)
        K = len(S_list)
        if K < 4:
            return True
        S_set = set(S_list)
        for idx_a, a in enumerate(S_list):
            others = [x for x in S_list if x != a]
            # Check a does not divide sum of any 3 distinct others
            for i in range(len(others)):
                x = others[i]
                for j in range(i + 1, len(others)):
                    y = others[j]
                    for k in range(j + 1, len(others)):
                        z = others[k]
                        s = x + y + z
                        if s % a == 0:
                            return False
        return True

    def can_add(a, S):
        """Check if adding a to S keeps validity. O(K^3) but with early exit."""
        S_list = list(S)
        K = len(S_list)
        S_set = set(S_list)

        # Check 1: a does not divide sum of any 3 distinct elements in S
        for i in range(K):
            x = S_list[i]
            for j in range(i + 1, K):
                y = S_list[j]
                for k in range(j + 1, K):
                    z = S_list[k]
                    if (x + y + z) % a == 0:
                        return False

        # Check 2: For each element b in S, b does not divide (a + u + v)
        # where u, v are 2 distinct elements from S \ {b}
        for b in S_list:
            # Triples involving a: {a, u, v} where u,v in S\{b}, u != v
            # Sum = a + u + v, need b ∤ (a + u + v)
            others_no_b = [x for x in S_list if x != b]
            for i in range(len(others_no_b)):
                u = others_no_b[i]
                for j in range(i + 1, len(others_no_b)):
                    v = others_no_b[j]
                    if (a + u + v) % b == 0:
                        return False
        return True

    # Strategy: Start with upper half [97, 192] greedily, then add smaller numbers
    S = set()
    
    # Phase 1: Greedily add from 192 down to 97
    for x in range(N, 96, -1):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)

    # Phase 2: Try to add smaller numbers
    for x in range(96, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)

    best_S = set(S)
    best_size = len(S)

    # Local search with simulated annealing-like approach
    rng = random.Random(12345)
    
    # Precompute all elements not in S
    all_elements = set(range(1, N + 1))
    
    while time.time() < deadline - 3.0:
        S_list = list(S)
        U = list(all_elements - S)
        
        if not U:
            break
        
        # Try random additions
        if len(S) < 48:
            # Try to add a random element from U
            for _ in range(50):
                if time.time() > deadline:
                    break
                a = rng.choice(U)
                if can_add(a, S):
                    S.add(a)
                    U.remove(a)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    if len(S) >= 48:
                        break
        
        # Try swaps: remove one, add one (or two)
        if S_list:
            r = rng.choice(S_list)
            S_mid = S - {r}
            U_mid = list(all_elements - S_mid)
            for _ in range(30):
                if time.time() > deadline:
                    break
                a = rng.choice(U_mid)
                if can_add(a, S_mid):
                    S = S_mid | {a}
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    break
            else:
                # Try double swap: remove two, add two
                if len(S_list) >= 2:
                    r2 = rng.choice([x for x in S_list if x != r])
                    S_mid2 = S - {r, r2}
                    U_mid2 = list(all_elements - S_mid2)
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        a1 = rng.choice(U_mid2)
                        if can_add(a1, S_mid2):
                            S_temp = S_mid2 | {a1}
                            U_mid2.remove(a1)
                            a2 = rng.choice(U_mid2)
                            if can_add(a2, S_temp):
                                S = S_temp | {a2}
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = set(S)
                                break
                    else:
                        S = S_mid2 | {r, r2}  # revert
                else:
                    S = S_mid2 | {r} if len(S_list) == 1 else S

    # Final verification
    if not is_valid_set(best_S):
        # Fallback: rebuild
        S = set()
        for x in range(N, 0, -1):
            if can_add(x, S):
                S.add(x)
        best_S = S

    return list(best_S)
