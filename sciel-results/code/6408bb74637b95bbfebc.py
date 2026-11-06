"""Rewritten program: find maximum clique in graph where nodes are [1,255] and edges connect numbers sharing a factor > 1."""
import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # All numbers in [1, 255]
    n = 255
    nodes = list(range(1, 256))

    # Build adjacency: adj[i] is a bitmask of nodes adjacent to node i
    # Two nodes are adjacent if gcd(nodes[i], nodes[j]) > 1
    adj = [0] * n
    for i in range(n):
        vi = nodes[i]
        for j in range(i + 1, n):
            if math.gcd(vi, nodes[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # We need a clique that includes 255 (index 254)
    # Start with the set of all nodes adjacent to 255 (plus 255 itself)
    # Then find max clique in the subgraph induced by those neighbors + 255

    # Actually, let's just find the max clique in the full graph that contains node 254 (which is 255)
    # We can do this by:
    # 1. Find all nodes adjacent to 255 (i.e., gcd(x, 255) > 1)
    # 2. Add 255 itself
    # 3. Find max clique in the induced subgraph on these nodes

    # Nodes that share a factor with 255 = 3*5*17
    # These are multiples of 3, 5, or 17 in [1,255]
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    # 255 is in candidates since gcd(255,255)=255>1
    # Map original indices to candidate indices
    cand_set = set(candidates)
    # Build index mapping
    orig_to_cand = {}
    for idx, val in enumerate(candidates):
        orig_to_cand[val] = idx

    m = len(candidates)

    # Build adjacency within candidates
    cand_adj = [0] * m
    for i in range(m):
        vi = candidates[i]
        for j in range(i + 1, m):
            if math.gcd(vi, candidates[j]) > 1:
                cand_adj[i] |= (1 << j)
                cand_adj[j] |= (1 << i)

    # Bron-Kerbosch with pivot to find maximum clique
    best_clique = []

    def popcount(x):
        return bin(x).count('1')

    def bron_kerbosch(R, P, X):
        """R, P, X are bitmasks over m nodes."""
        nonlocal best_clique

        if time.time() > DEADLINE:
            return

        if P == 0 and X == 0:
            if popcount(R) > len(best_clique):
                best_clique = [candidates[i] for i in range(m) if (R >> i) & 1]
            return

        # Choose pivot u from P | X to minimize branching
        union = P | X
        if union == 0:
            if popcount(R) > len(best_clique):
                best_clique = [candidates[i] for i in range(m) if (R >> i) & 1]
            return

        # Pick pivot with max degree in P
        best_u = -1
        best_deg = -1
        temp = union
        while temp:
            # Get lowest set bit
            lb = temp & (-temp)
            u = lb.bit_length() - 1
            deg = popcount(P & cand_adj[u])
            if deg > best_deg:
                best_deg = deg
                best_u = u
            temp ^= lb

        # Branch on candidates not adjacent to pivot
        candidates_to_try = P & ~cand_adj[best_u]
        temp = candidates_to_try
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            temp ^= lb

            # R ∪ {v}, P ∩ N(v), X ∩ N(v)
            bron_kerbosch(R | lb, P & cand_adj[v], X & cand_adj[v])
            P ^= lb
            X |= lb

    # Initial call: R=0, P=all candidates, X=0
    all_mask = (1 << m) - 1
    bron_kerbosch(0, all_mask, 0)

    # Ensure 255 is in the result (it should be since we're looking for max clique in the subgraph containing 255)
    # But let's verify and if not, add it (it should be there)
    if 255 not in best_clique:
        best_clique.append(255)

    # Verify the result
    result = best_clique
    # Check that 255 is in result
    if 255 not in result:
        result.append(255)

    # Verify all pairs have gcd > 1
    # If verification fails, fall back to a safe set
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: all multiples of 3 in [1,255] that are <= 255, plus ensure 255 is included
        # Multiples of 3: 3, 6, 9, ..., 255
        result = list(range(3, 256, 3))
        if 255 not in result:
            result.append(255)

    return result
