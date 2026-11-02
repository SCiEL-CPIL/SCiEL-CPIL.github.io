"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # The problem: cover all q^n words with balls of radius r.
    # For q=3, n=8, r=1, we want to minimize the number of balls.

    # Precompute: for each possible word (as an integer in 0..q^n-1),
    # compute the set of words within Hamming distance <= radius.
    # But that's 6561 words each with ~1 + 8*2 = 17 neighbors (for r=1, q=3).
    # Total: 6561 * 17 ~ 111k pairs. Manageable.

    # Convert between integer representation and word representation.
    def to_word(x):
        w = [0] * n
        for i in range(n):
            w[i] = x % q
            x //= q
        return w

    def to_int(w):
        x = 0
        for i in range(n - 1, -1, -1):
            x = x * q + w[i]
        return x

    total_words = q ** n

    # Build the ball for each word: all words within Hamming distance <= radius.
    # For efficiency, precompute all balls as sets of integers.
    # But storing 6561 sets of ~17 elements each is fine.

    # Precompute neighbor offsets for each position.
    # For position i, the offset to change digit i from d to d' is (d' - d) * q^i.
    # Precompute q^i for each position.
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute all words as integers
    all_words = list(range(total_words))

    # For each word, compute its ball (set of words within distance <= radius)
    # For radius=1: the word itself + all single-position changes.
    # For general radius, this is more complex. Let's handle radius=1 specifically
    # since that's the common case, and fall back to a simpler approach otherwise.

    if radius == 1:
        # For each word x, its ball is {x} ∪ {x + (d'-d)*q^i : i in 0..n-1, d' != d_i}
        # Precompute for each position i and each digit value d, the list of offsets.
        # Actually, let's just compute the ball for each word on the fly but cached.

        # Build balls as numpy arrays for speed
        # ball[x] = list of words in the ball centered at x
        balls = [None] * total_words
        for x in range(total_words):
            w = to_word(x)
            ball = [x]
            for i in range(n):
                d = w[i]
                for d_prime in range(q):
                    if d_prime != d:
                        y = x + (d_prime - d) * powers[i]
                        ball.append(y)
            balls[x] = ball

        # Now we have a set cover problem:
        # Universe: all_words (6561 elements)
        # Sets: balls[x] for x in 0..6560
        # We want to select the minimum number of sets that cover the universe.

        # Greedy set cover with shadow prices:
        # - Maintain for each word in the universe: how many selected balls cover it (should be 0 initially)
        # - Maintain for each candidate ball: how many uncovered words it covers
        # - Shadow price approach: prioritize balls that cover words with few remaining coverers

        # Let's use a different approach: 
        # For each word u in the universe, count how many balls contain u.
        # This is the "degree" of u. Words with low degree are "hard" to cover.
        # When selecting a ball, prefer balls that cover hard words.

        # Precompute: for each word u, the list of balls that contain u.
        # ball_containing[u] = list of x such that u is in balls[x]
        # This is the reverse mapping.

        # For radius=1, u is in balls[x] iff Hamming distance between u and x <= 1.
        # So ball_containing[u] = balls[u] (by symmetry of Hamming distance).

        # So the degree of u is |balls[u]| = 1 + n*(q-1) for all u.
        # That's constant! So all words have the same degree.
        # Shadow prices won't differentiate.

        # Let's try a different structural approach:
        # Use a greedy algorithm that at each step picks the ball that covers the most
        # uncovered words, with a tie-breaking based on some heuristic.

        # Track coverage: covered[u] = True if u is covered by at least one selected ball.
        # For each ball x, track how many uncovered words it covers.

        covered = np.zeros(total_words, dtype=bool)
        # For each ball, the count of uncovered words it covers
        # Initially, ball x covers len(balls[x]) words, all uncovered.
        # But we need to update this as words get covered.

        # This is O(total_words * ball_size) per iteration if done naively.
        # total_words = 6561, ball_size = 17, iterations up to ~486.
        # 486 * 6561 * 17 ~ 54M operations. Might be slow in Python.

        # Let's use numpy for speed.
        # Represent coverage as a boolean array.
        # For each candidate ball, compute the number of uncovered words it covers.

        # Precompute balls as a list of numpy arrays
        balls_np = [np.array(b, dtype=np.int32) for b in balls]

        # Greedy set cover
        selected = []
        uncovered_count = total_words

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            # Find the ball that covers the most uncovered words
            best_x = -1
            best_count = -1

            # Iterate over all candidate balls
            for x in range(total_words):
                if time.time() > deadline:
                    break
                # Count uncovered words in balls[x]
                ball_arr = balls_np[x]
                count = np.sum(~covered[ball_arr])
                if count > best_count:
                    best_count = count
                    best_x = x
                if best_count == 0:
                    break

            if best_x < 0 or best_count <= 0:
                # No ball covers any uncovered word; this shouldn't happen
                break

            # Select ball best_x
            selected.append(best_x)
            # Update covered
            covered[balls_np[best_x]] = True
            uncovered_count -= best_count

        # Convert selected integers to words
        result = [to_word(x) for x in selected]

        # Check if we have a valid cover; if not, add more balls
        # Verify coverage
        if not np.all(covered):
            # Find uncovered words and add balls for them
            uncovered_indices = np.where(~covered)[0]
            for u in uncovered_indices:
                if time.time() > deadline:
                    break
                # Add the ball centered at u
                if u not in selected:
                    selected.append(u)
                    covered[balls_np[u]] = True

            result = [to_word(x) for x in selected]

        return result

    else:
        # General radius: use a simpler greedy approach
        # For each word, compute its ball using BFS or combinatorial generation.
        # This is expensive for large n, but n=8, q=3 is manageable.

        def compute_ball(x, radius):
            w = to_word(x)
            ball = set()
            # Generate all words within Hamming distance <= radius
            # Use itertools combinations
            import itertools
            ball.add(x)
            for r in range(1, radius + 1):
                for positions in itertools.combinations(range(n), r):
                    for changes in itertools.product(range(q), repeat=r):
                        if any(c == w[p] for c, p in zip(changes, positions)):
                            continue
                        y = x
                        for c, p in zip(changes, positions):
                            y += (c - w[p]) * powers[p]
                        ball.add(y)
            return list(ball)

        # Build all balls
        balls = [compute_ball(x, radius) for x in range(total_words)]
        balls_np = [np.array(b, dtype=np.int32) for b in balls]

        covered = np.zeros(total_words, dtype=bool)
        uncovered_count = total_words
        selected = []

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            best_x = -1
            best_count = -1

            for x in range(total_words):
                if time.time() > deadline:
                    break
                ball_arr = balls_np[x]
                count = np.sum(~covered[ball_arr])
                if count > best_count:
                    best_count = count
                    best_x = x

            if best_x < 0 or best_count <= 0:
                break

            selected.append(best_x)
            covered[balls_np[best_x]] = True
            uncovered_count -= best_count

        result = [to_word(x) for x in selected]

        if not np.all(covered):
            uncovered_indices = np.where(~covered)[0]
            for u in uncovered_indices:
                if time.time() > deadline:
                    break
                if u not in selected:
                    selected.append(u)
                    covered[balls_np[u]] = True

            result = [to_word(x) for x in selected]

        return result
# EVOLVE-BLOCK-END
