import time
import itertools
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312"

    # Helper to build standard superpermutation
    def build_standard(n):
        word = [1]
        for m in range(2, n + 1):
            if time.time() > deadline:
                break
            perms, seen = [], set()
            w_size = m - 1
            for i in range(len(word) - w_size + 1):
                window = tuple(word[i:i + w_size])
                if len(set(window)) == w_size:
                    if window not in seen:
                        seen.add(window)
                        perms.append(window)

            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            word = new_word
        return "".join(map(str, word))

    # Standard construction for n
    if n <= 5:
        return build_standard(n)

    # For n >= 6, try to optimize the base
    # If n==6, we can search for a better one.
    # If n==7, we use the best n=6 we can find.

    best_n6 = None
    if n >= 6:
        # Try to find a better S6
        # A* search for n=6
        # State: (current_window, visited_edges_set)
        # This is too large for A* in Python if we track all visited edges.
        # Instead, use the standard construction for n=6 as a baseline.
        # Given the complexity, we might just stick to standard for now 
        # unless we can implement a very fast local search.

        # Let's try a simple local search on the standard S6
        s6_std = build_standard(6)
        best_s6 = s6_std

        # Local search: try to remove symbols
        # This is heuristic and might not find the 867 bound.
        # Given the time, we will return the standard construction 
        # but optimized for speed.

        if n == 6:
            return best_s6
        else:
            # Use best_s6 to build n=7
            # Modify build_standard to accept a base word?
            # The standard build_standard starts from 1 and builds up.
            # We can modify it to start from best_s6 for n=6 and build to 7.

            def build_from_base(base_word_str, target_n):
                word = list(map(int, base_word_str))
                current_n = 6 # Assuming base is for n=6
                for m in range(current_n + 1, target_n + 1):
                    if time.time() > deadline:
                        break
                    perms, seen = [], set()
                    w_size = m - 1
                    for i in range(len(word) - w_size + 1):
                        window = tuple(word[i:i + w_size])
                        if len(set(window)) == w_size:
                            if window not in seen:
                                seen.add(window)
                                perms.append(window)

                    new_word = []
                    for p in perms:
                        block = list(p) + [m] + list(p)
                        max_overlap = min(len(new_word), len(block))
                        overlap = 0
                        for o in range(max_overlap, -1, -1):
                            if o == 0:
                                overlap = 0
                                break
                            if new_word[len(new_word) - o:] == block[:o]:
                                overlap = o
                                break
                        new_word += block[overlap:]
                    word = new_word
                return "".join(map(str, word))

            return build_from_base(best_s6, n)

    return build_standard(n)
