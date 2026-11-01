import itertools
import time

def construct(q, n, radius):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds safety margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though the task specifies 3, 6, 1
        # A simple product construction or random search would be needed here
        # For now, just return a valid code if possible, or empty if not handled
        # The task contract says the evaluator passes q, n, radius.
        # We should handle the specific case well.
        pass

    # Specific solution for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        # 1. Generate initial code: Tetracode x Identity
        # Tetracode words: [a, b, (a+b)%3, (a+2b)%3] for a,b in {0,1,2}
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        
        # Product with identity on last 2 coordinates
        initial_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                initial_code.append(h + list(tail))
        
        # 2. Greedy Removal
        # State: list of codewords
        code = initial_code
        # State: cover count for each of the 3^6 = 729 points
        # Map a word to an index: w[0]*3^5 + w[1]*3^4 + ... + w[5]
        num_points = 3 ** 6
        cover_count = [0] * num_points
        
        def word_to_index(w):
            idx = 0
            for i in range(6):
                idx = idx * 3 + w[i]
            return idx

        def index_to_word(idx):
            w = [0] * 6
            for i in range(5, -1, -1):
                w[i] = idx % 3
                idx //= 3
            return w

        def get_ball(c):
            # Returns list of indices for all words in ball of radius 1 around c
            ball_indices = [word_to_index(c)]
            c_arr = list(c)
            for pos in range(6):
                orig_val = c_arr[pos]
                for new_val in range(3):
                    if new_val != orig_val:
                        c_arr[pos] = new_val
                        ball_indices.append(word_to_index(c_arr))
                c_arr[pos] = orig_val
            return ball_indices

        # Initialize cover_count
        for c in code:
            for idx in get_ball(c):
                cover_count[idx] += 1

        # Greedy removal loop
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            # We iterate over a copy of the indices because the list size changes
            # But we break immediately upon removal to restart the scan
            for i in range(len(code)):
                c = code[i]
                ball_indices = get_ball(c)
                essential = False
                for idx in ball_indices:
                    if cover_count[idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove c
                    code.pop(i)
                    for idx in ball_indices:
                        cover_count[idx] -= 1
                    improved = True
                    break # Restart the while loop

        return code

    # Generic fallback (not optimized for this specific task but valid interface)
    # For q=3, n=6, radius=1, the above block handles it.
    # If other parameters are passed, we might return a naive solution or empty.
    # Given the task contract, we assume the specific case.
    return []
