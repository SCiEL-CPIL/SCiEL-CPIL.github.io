"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill

    def gen_e8():
        # E8 lattice minimal vectors
        vecs = []
        # Integer vectors: e_i +/- e_j for i < j
        for i, j in itertools.combinations(range(8), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 8
                    v[i], v[j] = si, sj
                    vecs.append(tuple(v))
        # Half-integer vectors: all coords +/- 1/2, sum even
        # 8 coords, k plus signs, 8-k minus signs
        # sum = (k - (8-k))/2 = k - 4, must be even => k even
        for k in (0, 2, 4, 6, 8):
            for signs in itertools.combinations(range(8), k):
                for rest in itertools.product((1, -1), repeat=8 - k):
                    v = []
                    sign_idx = 0
                    for i in range(8):
                        if i in signs:
                            v.append(1)
                        else:
                            v.append(rest[sign_idx])
                            sign_idx += 1
                    # Scale by 2 to keep integers for now? No, let's keep as floats or multiply by 2
                    # Actually, to avoid floats in the final array if possible, but mixed types are hard.
                    # Let's generate as tuples of floats or ints.
                    # E8 vectors are either integers or half-integers.
                    # If we keep them as is, we have mixed types.
                    # The task asks for integer vectors.
                    # "Construct nonzero integer vectors"
                    # So we must scale?
                    # If we scale E8 by 2, we get integer vectors.
                    # Norm becomes 2 * sqrt(2) = sqrt(8).
                    # Distances scale by 2.
                    # This is valid.
                    vecs.append(tuple(2 * x for x in v))
        return np.array(vecs, dtype=np.int64)

    def gen_d3():
        # D3 lattice minimal vectors: e_i +/- e_j for i < j in {1,2,3}
        vecs = []
        for i, j in itertools.combinations(range(3), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 3
                    v[i], v[j] = si, sj
                    vecs.append(tuple(v))
        return np.array(vecs, dtype=np.int64)

    # Generate E8 (240 vectors) embedded in first 8 dims
    e8 = gen_e8()
    e8_ext = np.zeros((len(e8), D), dtype=np.int64)
    e8_ext[:, :8] = e8

    # Generate D3 (12 vectors) embedded in last 3 dims
    d3 = gen_d3()
    d3_ext = np.zeros((len(d3), D), dtype=np.int64)
    d3_ext[:, 8:] = d3

    # Combine
    combined = np.vstack([e8_ext, d3_ext])

    # Verify validity (optional, but good for sanity check during development)
    # Norms should all be equal to sqrt(8) or sqrt(2)?
    # E8 vectors scaled by 2: original norm sqrt(2) -> new norm sqrt(8).
    # D3 vectors: original norm sqrt(2) -> new norm sqrt(2).
    # Wait, the norms are different!
    # Task: "minimum pairwise distance is at least the maximum vector norm"
    # Max norm is sqrt(8) = 2*sqrt(2).
    # Min distance within E8 part: original min dist sqrt(2) -> new min dist sqrt(8) = 2*sqrt(2).
    # Min distance within D3 part: original min dist sqrt(2) -> new min dist sqrt(2).
    # Distance between E8 and D3 parts: sqrt(8 + 2) = sqrt(10).
    # Is min distance (sqrt(2)) >= max norm (sqrt(8))?
    # sqrt(2) ~ 1.414, sqrt(8) ~ 2.828.
    # 1.414 < 2.828. INVALID.
    
    # I made a mistake in the scaling logic.
    # To make it valid, all vectors must have the same norm, or at least the minimum distance must be >= max norm.
    # In a kissing configuration derived from a lattice, we usually take all minimal vectors.
    # If we take E8 minimal vectors (norm sqrt(2)) and D3 minimal vectors (norm sqrt(2)),
    # they have the same norm.
    # The distance between an E8 vector (in first 8 dims) and a D3 vector (in last 3 dims) is sqrt(||v_e8||^2 + ||v_d3||^2) = sqrt(2 + 2) = 2.
    # The distance between two E8 vectors is at least sqrt(2).
    # The distance between two D3 vectors is at least sqrt(2).
    # Max norm is sqrt(2).
    # Min distance is sqrt(2).
    # sqrt(2) >= sqrt(2). Valid.
    # So I should NOT scale E8 by 2. I should use the half-integers?
    # But the task requires INTEGER vectors.
    # So I MUST scale to integers.
    # If I scale E8 by 2, norm is sqrt(8).
    # If I keep D3 as is, norm is sqrt(2).
    # Mixed norms are a problem for "equal-radius spheres" implication, but the check is "min dist >= max norm".
    # If I scale D3 by 2 as well?
    # D3 scaled by 2: norm sqrt(8).
    # Then all norms are sqrt(8).
    # Min dist E8-E8: sqrt(8).
    # Min dist D3-D3: sqrt(8).
    # Min dist E8-D3: sqrt(8+8) = 4.
    # Max norm: sqrt(8) ~ 2.82.
    # Min dist: sqrt(8) ~ 2.82.
    # 2.82 >= 2.82. Valid.
    
    # So, let's scale BOTH by 2.
    # E8 vectors (integers or half-integers).
    # If half-integers, multiplying by 2 gives integers.
    # If integers, multiplying by 2 gives even integers.
    # Let's just generate the integer versions directly.
    
    def gen_e8_int():
        # Returns integer vectors in R^8 corresponding to E8 minimal vectors scaled by 2.
        vecs = []
        # Integer type: e_i +/- e_j. Scaled by 2: 2e_i +/- 2e_j.
        for i, j in itertools.combinations(range(8), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 8
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vecs.append(tuple(v))
        # Half-integer type: all coords +/- 1/2. Scaled by 2: all coords +/- 1.
        # Sum of original coords is even integer.
        # Original coords are +/- 1/2.
        # Let x_i be +/- 1. Sum x_i must be even?
        # Original sum = (sum x_i) / 2.
        # If sum x_i is even, original sum is integer.
        # E8 condition: sum is even integer.
        # So sum x_i must be even.
        for x in itertools.product((1, -1), repeat=8):
            if sum(x) % 2 == 0:
                vecs.append(tuple(x))
        return np.array(vecs, dtype=np.int64)

    def gen_d3_int():
        # D3 minimal vectors: e_i +/- e_j.
        # Scale by 2?
        # If we scale D3 by 2, we get 2e_i +/- 2e_j.
        vecs = []
        for i, j in itertools.combinations(range(3), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 3
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vecs.append(tuple(v))
        return np.array(vecs, dtype=np.int64)

    e8 = gen_e8_int()
    e8_ext = np.zeros((len(e8), D), dtype=np.int64)
    e8_ext[:, :8] = e8

    d3 = gen_d3_int()
    d3_ext = np.zeros((len(d3), D), dtype=np.int64)
    d3_ext[:, 8:] = d3

    combined = np.vstack([e8_ext, d3_ext])
    
    return combined
# EVOLVE-BLOCK-END
