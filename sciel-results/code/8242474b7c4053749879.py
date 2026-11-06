import numpy as np
import time
import random
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 55  # Safety margin before 60s limit

    # Precompute masks for blocking
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)

    def get_neighbors(w):
        # Yield all words within distance < MIN_DIST
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                yield w ^ mask

    # 1. Find a linear code [17, 8, 6]
    k = 8
    linear_code = None

    # Search for linear code
    max_linear_search_time = 8.0
    while time.time() - start_time < max_linear_search_time:
        G = np.random.randint(0, 2, size=(k, N))

        # Convert G rows to integers
        G_ints = []
        for i in range(k):
            val = 0
            for j in range(N):
                if G[i, j] == 1:
                    val |= (1 << j)
            G_ints.append(val)

        # Generate all 2^k codewords
        codewords = [0]
        for i in range(k):
            new_codewords = [c ^ G_ints[i] for c in codewords]
            codewords.extend(new_codewords)

        # Check min distance (for linear codes, min weight of non-zero codewords = min distance)
        min_d = N + 1
        for c in codewords:
            if c == 0:
                continue
            w = bin(c).count('1')
            if w < min_d:
                min_d = w
                if min_d < MIN_DIST:
                    break
        if min_d >= MIN_DIST:
            linear_code = codewords
            break

    if linear_code is None:
        # Fallback: Simple greedy from 0
        blocked = bytearray(1 << N)
        code = []
        def add_word(w):
            if blocked[w]:
                return
            code.append(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1

        add_word(0)
        for w in range(1, 1 << N):
            if time.time() - start_time > time_limit:
                break
            if not blocked[w]:
                add_word(w)
        return code

    # 2. Expand the linear code using greedy + local search
    blocked = bytearray(1 << N)
    code = list(linear_code)  # 256 words

    # Block all neighbors of the linear code
    for w in code:
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                blocked[w ^ mask] = 1

    # Greedy scan for additional words
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if not blocked[w]:
            code.append(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1

    # 3. Local search: try to improve by removing one word and re-greedy'ing
    best_code = list(code)
    best_size = len(code)

    # Build a set for fast membership testing
    code_set = set(code)

    # Try multiple passes of local search
    local_search_deadline = time_limit - 5  # Leave 5s safety margin

    while time.time() - start_time < local_search_deadline:
        improved = False

        # Try removing each word and see if we can add more
        for idx in range(len(code)):
            if time.time() - start_time > local_search_deadline:
                break

            # Remove word at idx
            removed_word = code[idx]
            code.pop(idx)
            code_set.remove(removed_word)

            # Unblock its neighbors
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[removed_word ^ mask] = 0

            # Re-greedy: try to add words that were blocked
            new_code = list(code)
            new_code_set = set(code)

            # Scan all words, add if not blocked
            for w in range(1 << N):
                if time.time() - start_time > local_search_deadline:
                    break
                if not blocked[w] and w not in new_code_set:
                    new_code.append(w)
                    new_code_set.add(w)
                    for d in range(MIN_DIST):
                        for mask in masks_by_dist[d]:
                            blocked[w ^ mask] = 1

            # Check if we improved
            if len(new_code) > best_size:
                best_size = len(new_code)
                best_code = new_code
                improved = True
                # Update current state
                code = new_code
                code_set = new_code_set
                break  # Restart the loop with improved code

            # Restore if no improvement
            code.insert(idx, removed_word)
            code_set.add(removed_word)
            # Re-block its neighbors
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[removed_word ^ mask] = 1

        if not improved:
            break

    return best_code
