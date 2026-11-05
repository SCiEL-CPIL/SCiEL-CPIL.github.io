import itertools
import time
import sys
import random

def construct(n):
    # Deadline guard
    start_time = time.time()
    timeout = 115.0  # 5 sec margin

    # Classic construction
    def classic_superpermutation(n):
        if n == 1:
            return [1]
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return word

    # Get classic result
    classic_word = classic_superpermutation(n)
    classic_len = len(classic_word)

    # For small n, classic is optimal
    if n <= 5:
        return "".join(map(str, classic_word))

    # Try Simulated Annealing for n=6, 7
    if n in [6, 7]:
        try:
            # Generate all permutations
            perms = list(itertools.permutations(range(1, n + 1)))
            N = len(perms)

            # Precompute overlap matrix: overlap[i][j] = max k such that
            # suffix of perms[i] of length k == prefix of perms[j] of length k
            # We only need suffix-prefix overlaps
            overlap = [[0] * N for _ in range(N)]

            # Precompute for faster overlap computation
            # For each permutation, precompute its suffixes and prefixes
            # Actually, let's precompute prefix maps for each length
            prefix_maps = [None] * (n + 1)
            for k in range(1, n):
                pm = {}
                for i, p in enumerate(perms):
                    pref = p[:k]
                    if pref not in pm:
                        pm[pref] = []
                    pm[pref].append(i)
                prefix_maps[k] = pm

            # Compute overlap matrix
            for i in range(N):
                curr = perms[i]
                for k in range(n - 1, 0, -1):
                    suffix = curr[n - k:]
                    # Find all j where prefix of perms[j] of length k == suffix
                    # We can use prefix_maps[k]
                    # But we need to be careful: we want the first match? No, we want max overlap
                    # Actually for a fixed i and k, we want to find if there exists j with prefix_k(j) == suffix_k(i)
                    # The overlap[i][j] is the max k such that suffix_k(i) == prefix_k(j)
                    # Let's compute it differently: for each pair, find max k
                    # This is O(N^2 * n) which is 5040^2 * 7 ≈ 178M operations - too slow in Python
                    # Let's use the prefix maps approach
                    pass

            # Better approach: for each i, for k from n-1 down to 1,
            # check if suffix_k(i) is in prefix_maps[k], and if so, mark overlap[i][j] = k for those j
            # But we need max k, so we go from large k to small k and only set if not already set
            # Actually, overlap[i][j] should be the MAX k. So for each i, we iterate k from n-1 to 1,
            # and for each j found, if overlap[i][j] is still 0, set it to k.
            # But this gives us for each (i,j) the max k where suffix_k(i) == prefix_k(j).
            # Wait, that's not quite right. overlap[i][j] = max k such that suffix_k(i) == prefix_k(j).
            # So for fixed i, we want for each j, the max k.
            # We can do: for k from n-1 down to 1, for each j in prefix_maps[k][suffix_k(i)], if overlap[i][j] == 0, set overlap[i][j] = k.
            # This works because we check larger k first.

            for i in range(N):
                curr = perms[i]
                for k in range(n - 1, 0, -1):
                    suffix = curr[n - k:]
                    candidates = prefix_maps[k].get(suffix, [])
                    for j in candidates:
                        if overlap[i][j] == 0:
                            overlap[i][j] = k

            # Now define energy function
            # Energy = sum over adjacent pairs (a, b) of (n - overlap[a][b])
            # Because if we place permutation a followed by b, we add n - overlap[a][b] new symbols
            # Total length = n + sum_{i=0}^{N-2} (n - overlap[order[i]][order[i+1]])
            # = n + (N-1)*n - sum_{i=0}^{N-2} overlap[order[i]][order[i+1]]
            # = N*n - sum_{i=0}^{N-2} overlap[order[i]][order[i+1]]
            # So minimizing length = maximizing sum of overlaps

            def compute_energy(order):
                """Compute total length given an ordering of permutations."""
                total = n
                for i in range(N - 1):
                    total += n - overlap[order[i]][order[i + 1]]
                return total

            # Initial ordering: use greedy construction
            def greedy_order():
                visited = [False] * N
                order = [0]
                visited[0] = True
                current = 0
                for _ in range(N - 1):
                    best_j = -1
                    best_ov = 0
                    for k in range(n - 1, 0, -1):
                        suffix = perms[current][n - k:]
                        candidates = prefix_maps[k].get(suffix, [])
                        for j in candidates:
                            if not visited[j]:
                                best_j = j
                                best_ov = k
                                break
                        if best_j != -1:
                            break
                    if best_j == -1:
                        for j in range(N):
                            if not visited[j]:
                                best_j = j
                                best_ov = 0
                                break
                    visited[best_j] = True
                    order.append(best_j)
                    current = best_j
                return order

            # Try to find a valid ordering with greedy
            # But greedy might get stuck. Let's just start with a random permutation
            # and use simulated annealing.

            # Simulated Annealing
            random.seed(42)
            current_order = list(range(N))
            random.shuffle(current_order)
            current_energy = compute_energy(current_order)
            best_order = current_order[:]
            best_energy = current_energy

            # Annealing schedule
            T_start = 10.0
            T_end = 0.01
            alpha = 0.999  # cooling rate
            T = T_start

            # Run until time limit
            steps_per_sec_target = 50000  # target ~50k steps per second

            while time.time() - start_time < timeout - 5.0:
                # Pick two random positions to swap
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    continue

                # Compute energy change for swapping order[i] and order[j]
                # Only adjacent pairs are affected
                # Affected positions: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
                # But need to be careful if i and j are adjacent

                old_a = current_order[i]
                old_b = current_order[j]

                # Remove contributions from affected edges
                delta = 0
                # Edge before i
                if i > 0:
                    prev = current_order[i - 1]
                    delta += (n - overlap[prev][old_a])
                # Edge after i
                if i < N - 1:
                    next_pos = current_order[i + 1]
                    delta += (n - overlap[old_a][next_pos])
                # Edge before j
                if j > 0:
                    prev = current_order[j - 1]
                    delta += (n - overlap[prev][old_b])
                # Edge after j
                if j < N - 1:
                    next_pos = current_order[j + 1]
                    delta += (n - overlap[old_b][next_pos])

                # Perform swap
                current_order[i], current_order[j] = old_b, old_a

                # Add new contributions
                # Edge before i
                if i > 0:
                    prev = current_order[i - 1]
                    delta -= (n - overlap[prev][old_b])
                # Edge after i
                if i < N - 1:
                    next_pos = current_order[i + 1]
                    delta -= (n - overlap[old_b][next_pos])
                # Edge before j
                if j > 0:
                    prev = current_order[j - 1]
                    delta -= (n - overlap[prev][old_a])
                # Edge after j
                if j < N - 1:
                    next_pos = current_order[j + 1]
                    delta -= (n - overlap[old_a][next_pos])

                new_energy = current_energy + delta

                # Accept if better or with probability exp(-delta/T)
                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_order = current_order[:]
                else:
                    # Accept with probability exp(-delta/T)
                    if T > 0:
                        import math
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            current_energy = new_energy
                        else:
                            # Reject swap
                            current_order[i], current_order[j] = old_a, old_b

                # Cool down
                T *= alpha
                if T < T_end:
                    T = T_end

            # Build final word from best_order
            def build_word(order):
                word = list(perms[order[0]])
                for i in range(N - 1):
                    ov = overlap[order[i]][order[i + 1]]
                    word.extend(perms[order[i + 1]][ov:])
                return word

            best_word = build_word(best_order)
            best_len = len(best_word)

            if best_len < classic_len:
                return "".join(map(str, best_word))

            # If SA didn't improve, return classic
            return "".join(map(str, classic_word))

        except Exception:
            pass

    return "".join(map(str, classic_word))
