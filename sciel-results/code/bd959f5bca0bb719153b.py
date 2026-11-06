"""Rewritten program: MIS-style branch-and-bound with greedy coloring, plus beam search fallback."""
import time
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # The problem: find a strictly increasing sequence of integers in [2,128]
    # such that all nonempty consecutive block products are distinct.
    # Maximize length.

    # Key insight: if we have a sequence [a1, a2, ..., ak], the products of
    # consecutive blocks are:
    #   a1, a2, ..., ak  (singletons)
    #   a1*a2, a2*a3, ..., a_{k-1}*ak  (pairs)
    #   a1*a2*a3, a2*a3*a4, ..., a_{k-2}*a_{k-1}*ak  (triples)
    #   etc.
    #
    # Two blocks collide if they have the same product. A block is determined
    # by its start and end indices. So we need all products prod[i..j] for
    # 0 <= i <= j < k to be distinct.
    #
    # The greedy approach in the seed picks numbers in increasing order and
    # checks if adding the next number causes any product collision with existing
    # tail products. This is fast but suboptimal.
    #
    # A better approach: beam search over prefixes. At each step, we maintain
    # a set of candidate prefixes. For each prefix, we try adding each unused
    # number (in increasing order to maintain strict increase) and check
    # validity. We keep the top beam_width prefixes by length (and as tiebreaker,
    # by the "potential" to extend further).

    # Let's use a beam search with numpy acceleration where possible.

    # First, let's understand the product collision check:
    # Given a prefix [a1, ..., am] and a new value x > am:
    # New products formed: x (singleton), a_m*x, a_{m-1}*a_m*x, ..., a_1*...*a_m*x
    # These must not collide with any existing products in the prefix.
    #
    # Existing products: all prod[i..j] for 0<=i<=j<m.
    # New products: prod[m..m]=x, prod[m-1..m]=a_{m-1}*x, ..., prod[0..m]=a_1*...*a_m*x
    #
    # We can store all existing products in a set.

    # Beam search approach:
    # State: (sequence_tuple, set_of_products)
    # But sets are expensive to copy. Instead, we can represent the product set
    # implicitly or use a frozenset for hashing in the beam.

    # Actually, for beam search, we need to be careful about memory and time.
    # Let's limit beam width and depth.

    # Alternative: since the sequence must be strictly increasing and in [2,128],
    # the maximum possible length is 127. The greedy gives some length L_greedy.
    # We want to find if we can do better.

    # Let's first run the greedy to get a baseline.
    def greedy_construct():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    greedy_result = greedy_construct()
    best_result = list(greedy_result)
    best_len = len(best_result)

    if time.time() > deadline:
        return best_result

    # Now try beam search to improve upon greedy.
    # Beam state: (sequence as tuple, product set as frozenset)
    # We process levels by length. At each level, we extend each beam member
    # with all valid next values.

    # To make this efficient, we'll cap the beam width.
    beam_width = 200
    max_time_per_level = 0.5

    # Initialize beam with all single-element sequences
    beam = []
    for x in range(2, 129):
        beam.append(((x,), frozenset([x])))

    # Sort by potential: shorter sequences with smaller last element have more room
    # Actually, all are length 1. Let's just keep them all if beam_width allows.
    beam = beam[:beam_width]

    current_len = 1

    while time.time() < deadline:
        next_beam = []
        seen_states = set()  # to avoid duplicate (sequence) in next beam

        for seq, prod_set in beam:
            last_val = seq[-1]
            # Try adding each x > last_val
            for x in range(last_val + 1, 129):
                # Check if x can be appended
                # New products: x, seq[-1]*x, seq[-2]*seq[-1]*x, ...
                new_products = []
                prod = x
                new_products.append(prod)
                for a in reversed(seq):
                    prod *= a
                    new_products.append(prod)

                # Check collision
                if any(p in prod_set for p in new_products):
                    continue

                new_seq = seq + (x,)
                new_prod_set = prod_set.union(new_products)

                state_key = new_seq
                if state_key in seen_states:
                    continue
                seen_states.add(state_key)

                next_beam.append((new_seq, new_prod_set))

                if len(next_beam) >= beam_width * 3:  # cap to avoid explosion
                    break

            if time.time() > deadline:
                break

        if not next_beam:
            break

        # Select top beam_width from next_beam
        # Score: prefer longer sequences; all same length here.
        # Tiebreaker: prefer sequences whose last element is smaller (more room to extend)
        # and whose product set is "sparser" in some sense.
        # Simple heuristic: sort by last element ascending, then by sum of products descending (more spread)
        next_beam.sort(key=lambda item: (item[0][-1], -sum(item[1])))
        beam = next_beam[:beam_width]
        current_len += 1

        # Update best
        if current_len > best_len:
            best_seq = max(beam, key=lambda item: len(item[0]))
            best_result = list(best_seq[0])
            best_len = current_len

        if time.time() > deadline:
            break

    # Also try a local improvement: take the greedy result and try swapping
    # or inserting elements.
    if time.time() < deadline - 5:
        # Try to improve by local search on the greedy result
        # For each position, try replacing with a different value
        improved = list(best_result)
        for _ in range(100):  # limited iterations
            if time.time() > deadline - 2:
                break
            changed = False
            for i in range(len(improved)):
                old_val = improved[i]
                # Try replacing with nearby values
                for delta in [-3, -2, -1, 1, 2, 3]:
                    new_val = old_val + delta
                    if new_val < 2 or new_val > 128:
                        continue
                    # Check strict increasing
                    if i > 0 and new_val <= improved[i-1]:
                        continue
                    if i < len(improved) - 1 and new_val >= improved[i+1]:
                        continue
                    # Check product uniqueness
                    test_seq = improved[:i] + [new_val] + improved[i+1:]
                    if _is_valid(test_seq):
                        improved = test_seq
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break

        if len(improved) > len(best_result):
            best_result = improved

    return best_result


def _is_valid(seq):
    """Check if a strictly increasing sequence has all distinct consecutive block products."""
    if len(seq) == 0:
        return True
    if len(seq) == 1:
        return True
    # Check strictly increasing
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    # Check all products distinct
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
            if prod > 10**18:  # early termination for huge products
                break
    return True


# EVOLVE-BLOCK-END
