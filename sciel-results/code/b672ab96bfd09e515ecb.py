import time
import sys

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    def build_paley(p):
        """Build Paley tournament on p vertices (p must be prime, p % 4 == 3)."""
        squares = [False] * p
        for x in range(1, p):
            squares[(x * x) % p] = True

        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if squares[diff]:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    def check_valid(p, orientation):
        """Check if every 4-set has a common dominator using bitsets."""
        # Build in_neighbors as Python ints (arbitrary precision bitsets)
        in_neighbors = [0] * p
        idx = 0
        for i in range(p):
            for j in range(i + 1, p):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    # i -> j, so i is in in_neighbors[j]
                    in_neighbors[j] |= (1 << i)
                else:
                    # j -> i, so j is in in_neighbors[i]
                    in_neighbors[i] |= (1 << j)

        # Check all 4-subsets
        for a in range(p):
            for b in range(a + 1, p):
                ab = in_neighbors[a] & in_neighbors[b]
                for c in range(b + 1, p):
                    abc = ab & in_neighbors[c]
                    for d in range(c + 1, p):
                        if abc & in_neighbors[d] == 0:
                            return False
        return True

    # Candidate primes p % 4 == 3, starting from smallest
    candidates = [7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79]

    for p in candidates:
        if time.time() - start_time > deadline:
            break

        orientation = build_paley(p)

        if check_valid(p, orientation):
            return {'n': p, 'orientation': orientation}

    # Fallback: return the 67-vertex Paley tournament
    p = 67
    orientation = build_paley(p)
    return {'n': p, 'orientation': orientation}
