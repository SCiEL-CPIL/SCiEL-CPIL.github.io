"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _pale_tournament(p):
    """Return (n, orientation_list) for a Paley tournament on p vertices.
    orientation bit for pair (i,j), i<j: 0 if i->j, 1 if j->i.
    i->j iff (j-i) mod p is a quadratic residue (nonzero square)."""
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    n = p
    orient = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % p in squares:
                orient.append(0)
            else:
                orient.append(1)
    return n, orient


def _check_validity(n, orient):
    """Check if every 4-set has a common dominator.
    orient is a list of bits for pairs (i,j) i<j in lexicographic order.
    0 means i->j, 1 means j->i.
    A vertex v dominates a set S if v has arcs to all vertices in S.
    For a 4-set {a,b,c,d}, we need some v in {a,b,c,d} such that v beats all other 3.
    """
    # Build adjacency: adj[i][j] = True if i -> j
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orient[idx] == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True
            idx += 1

    # For each 4-set, check if any vertex dominates all others in the set
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    s = [a, b, c, d]
                    found = False
                    for v in s:
                        ok = True
                        for w in s:
                            if v != w and not adj[v][w]:
                                ok = False
                                break
                        if ok:
                            found = True
                            break
                    if not found:
                        return False
    return True


def _count_violations(n, orient):
    """Count number of 4-sets without a common dominator."""
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orient[idx] == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True
            idx += 1

    violations = 0
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    s = [a, b, c, d]
                    found = False
                    for v in s:
                        ok = True
                        for w in s:
                            if v != w and not adj[v][w]:
                                ok = False
                                break
                        if ok:
                            found = True
                            break
                    if not found:
                        violations += 1
    return violations


def _local_search(n, orient, max_time=100.0):
    """Try to improve a tournament by flipping edges to reduce violations."""
    start_time = time.time()
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orient[idx] == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True
            idx += 1

    def violations_fast(adj):
        """Count violations using adjacency matrix."""
        vcount = 0
        for a in range(n - 3):
            for b in range(a + 1, n - 2):
                for c in range(b + 1, n - 1):
                    for d in range(c + 1, n):
                        s = (a, b, c, d)
                        found = False
                        for v in s:
                            ok = True
                            for w in s:
                                if v != w and not adj[v][w]:
                                    ok = False
                                    break
                            if ok:
                                found = True
                                break
                        if not found:
                            vcount += 1
        return vcount

    current_viol = violations_fast(adj)
    if current_viol == 0:
        # Convert back to orient list
        result = []
        for i in range(n):
            for j in range(i + 1, n):
                result.append(0 if adj[i][j] else 1)
        return n, result

    best_n, best_orient = n, orient[:]
    best_viol = current_viol

    while time.time() - start_time < max_time:
        # Pick a random edge to flip
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)

        # Flip
        adj[i][j] = not adj[i][j]
        adj[j][i] = not adj[j][i]

        new_viol = violations_fast(adj)

        if new_viol < current_viol:
            current_viol = new_viol
            if new_viol < best_viol:
                best_viol = new_viol
                best_n = n
                best_orient = []
                for ii in range(n):
                    for jj in range(ii + 1, n):
                        best_orient.append(0 if adj[ii][jj] else 1)
        else:
            # Revert
            adj[i][j] = not adj[i][j]
            adj[j][i] = not adj[j][i]

        if current_viol == 0:
            break

    return best_n, best_orient


def construct():
    """Construct a tournament where every 4-set has a common dominator.
    Minimize n. Return {'n': n, 'orientation': [...]}."""
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit

    # Try Paley tournaments for primes p ≡ 3 (mod 4) where Paley tournaments are known
    # to have good properties. Smallest such primes: 3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83
    # We need n >= 5. Try from small to large.
    candidate_primes = [7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83]

    best_result = None

    for p in candidate_primes:
        if time.time() - start_time > deadline - 20:
            break
        n, orient = _pale_tournament(p)
        # Check validity
        if _check_validity(n, orient):
            best_result = {'n': n, 'orientation': orient}
            break  # Found smallest valid one

    if best_result is not None:
        return best_result

    # If no small Paley tournament works, try local search on the smallest prime
    # that might work. Start with p=31 (Erdos lower bound is 31).
    if time.time() - start_time < deadline - 10:
        for p in [31, 43, 47, 59, 67]:
            if time.time() - start_time > deadline - 5:
                break
            n, orient = _pale_tournament(p)
            n2, orient2 = _local_search(n, orient, max_time=min(10, deadline - (time.time() - start_time) - 5))
            if _check_validity(n2, orient2):
                return {'n': n2, 'orientation': orient2}

    # Fallback: return the known 83-vertex Paley tournament
    n, orient = _pale_tournament(83)
    return {'n': n, 'orientation': orient}

# EVOLVE-BLOCK-END
