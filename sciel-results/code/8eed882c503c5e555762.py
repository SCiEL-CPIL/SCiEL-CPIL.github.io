import time
import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Deadline guard: track elapsed time to ensure we return within limits.
    # Since priority is called per-vector in a sort, we can't do heavy work here.
    # However, we can use a simple, fast heuristic.
    
    # The goal is to maximize the size of the cap set.
    # A cap set in F_3^n is a set with no three distinct elements that sum to 0.
    # (i.e., no x, y, z in the set such that x+y+z = 0 mod 3).
    
    # Heuristic: Vectors that are "more random" or "more balanced" tend to be better.
    # However, simple balance (count of 0s, 1s, 2s) was the previous approach.
    
    # Let's try a different structural feature:
    # 1. Balance score (product of counts)
    # 2. Penalty for "near-AP" patterns.
    #    In F_3, an arithmetic progression of length 3 is x, x+d, x+2d.
    #    This is equivalent to x+y+z = 0 with distinct x,y,z.
    #    A vector with many repeated coordinates is "closer" to being part of such a structure?
    #    Actually, vectors with many distinct coordinates are generally "safer" in some heuristics,
    #    but balance is key.
    
    # Let's refine the balance score.
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # The product c0*c1*c2 is a good measure of balance.
    balance_score = float(c0 * c1 * c2)
    
    # Secondary term: Penalize vectors that have many pairs of equal coordinates.
    # Why? If a vector has many equal coordinates, it might be more likely to form
    # a line (AP) with other vectors that also have similar structures.
    # Specifically, if vector v has v[i] == v[j] for many pairs, then any vector w
    # with w[i] == w[j] and w[i]+w[j] = -2*v[i] (mod 3) might form an AP with v?
    # This is getting complicated. Let's stick to a simpler proxy.
    
    # A simpler proxy for "randomness" or "non-structure":
    # Count the number of distinct values. But we already have 0,1,2.
    # How about the variance of the coordinates?
    # In F_3, the "variance" isn't standard.
    
    # Let's try: Reward vectors where the coordinates are more "spread out" in a way
    # that avoids local patterns.
    # Actually, let's look at the inspiration: "penalize near-AP patterns (vectors with two equal coordinates)".
    # This suggests that vectors with all distinct coordinates (if n<=3) or fewer repeats are better.
    # For large n, we can count the number of "repeats".
    
    # Number of pairs (i,j) with i<j and vector[i] == vector[j].
    # This is sum over v in {0,1,2} of C(count_v, 2).
    repeats = 0
    for c in (c0, c1, c2):
        repeats += c * (c - 1) // 2
    
    # We want to minimize repeats? Or is it the other way around?
    # If we minimize repeats, we prefer vectors with balanced counts (since for fixed n,
    # the sum of C(c_v, 2) is minimized when c_v are as equal as possible).
    # So minimizing repeats is equivalent to maximizing balance.
    # So this is redundant with the balance score.
    
    # Let's try a different angle: Use a hash-based tie-breaker that is more "chaotic"
    # to avoid any subtle biases in the greedy algorithm.
    
    b = bytes(vector)
    h = zlib.crc32(b)
    
    # Combine balance and hash.
    # The balance score is an integer (or float).
    # The hash is a 32-bit integer.
    # We want the balance to dominate, but the hash to break ties.
    
    # If we just add a small fraction of the hash, it might not be enough to break ties
    # if the balance scores are close integers.
    # But balance scores are integers. So if two vectors have the same balance,
    # the hash will decide.
    
    # However, maybe we should use a more complex priority that isn't just balance.
    # What if we prioritize vectors that are "orthogonal" to previously added vectors?
    # We don't have access to the current set in the priority function.
    
    # Let's try a different heuristic:
    # Score = balance_score - alpha * (some measure of structure)
    # What structure? Maybe the number of "zero-sum" triples within the vector itself?
    # But a single vector doesn't form a triple.
    
    # Let's stick to the balance + hash, but maybe adjust the hash contribution.
    # The previous score was 1.0430. The inspirations got 1.0977.
    # The inspirations mentioned "structural spread/balance priority plus a CRC tie-break".
    
    # Let's try to make the tie-breaker more significant to ensure a truly random order
    # among balanced vectors.
    
    # Normalize the hash to [0, 1).
    tie_breaker = (h % 1000000) * 1e-6
    
    # Return balance_score + tie_breaker
    return balance_score + tie_breaker
# EVOLVE-BLOCK-END
