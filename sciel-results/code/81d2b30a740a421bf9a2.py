"""Structural redesign: exact subset-sum avoidance via bitset reachability + local search."""
import time

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    TARGET = 200
    N = 100

    # Bitset of reachable sums: bit i set iff sum i is achievable.
    # We cap at TARGET because we only care about sums <= TARGET.
    # mask: bits 0..TARGET inclusive
    mask = (1 << (TARGET + 1)) - 1

    # Greedy seed: add numbers from large to small if it doesn't create sum=TARGET
    # This gives a good initial solution.
    def greedy_seed():
        reachable = 1  # bit 0 set
        selected = []
        for x in range(N, 0, -1):
            new_bits = (reachable << x) & mask
            if not (new_bits >> TARGET) & 1:
                selected.append(x)
                reachable |= new_bits
        return selected, reachable

    selected, reachable = greedy_seed()
    best_size = len(selected)

    # Local search: try to add an excluded element, or swap to increase size
    # Excluded elements
    selected_set = set(selected)
    excluded = [x for x in range(1, N + 1) if x not in selected_set]

    # Try adding each excluded element
    def try_add(x, reachable):
        """Return new reachable if adding x doesn't create TARGET, else None."""
        new_bits = (reachable << x) & mask
        if (new_bits >> TARGET) & 1:
            return None
        return reachable | new_bits

    # Try to add as many as possible
    improved = True
    while improved and (time.time() - t_start) < DEADLINE:
        improved = False
        # Sort excluded by how "safe" they are (fewer conflicts)
        # For each excluded x, check if it can be added
        for i, x in enumerate(excluded):
            if (time.time() - t_start) > DEADLINE:
                break
            new_reach = try_add(x, reachable)
            if new_reach is not None:
                selected.append(x)
                selected_set.add(x)
                reachable = new_reach
                excluded.pop(i)
                improved = True
                best_size = len(selected)
                break  # restart scan after modification

    # Now try swaps: remove one selected, add up to two excluded to net +1
    # This is more expensive; do a few passes
    for _pass in range(5):
        if (time.time() - t_start) > DEADLINE:
            break
        found_swap = False
        for i, rem in enumerate(selected):
            if (time.time() - t_start) > DEADLINE:
                break
            # Compute reachable without rem
            # We need to recompute reachable from scratch minus rem
            # This is expensive; use a different approach:
            # Check if removing rem and adding two excluded gives +1 net
            # Skip for now due to cost; instead try: remove rem, then greedily add back
            pass

    # Alternative structural approach: use the fact that if we pick all numbers > 100... no, max is 100.
    # Key insight: any subset summing to 200 from [1,100]. 
    # Numbers 101..200 aren't in range. 
    # If we pick all numbers from 51 to 100 (50 numbers), the minimum subset sum is 51, 
    # and we need to check if any subset sums to 200.
    # 100+99+...+? Let's think: can we do better with a different structure?

    # Try: all numbers in (TARGET/2, N] = (100, 100] = just {100}? No, 51..100.
    # Sum of any two from 51..100 is at least 102, at most 200. 100+100 not allowed (once).
    # 100+99=199, 100+98=198... 100+51=151. Three smallest: 51+52+53=156. 
    # Can 51..100 have a subset summing to 200? Yes: 100+99+1? No, 1 not in set. 
    # 100+50? 50 not in set. 99+98+3? No. Let me check: 100+99=199, need 1 more - not available.
    # 100+98=198, need 2 - not available. ... 100+51=151, need 49 - not available.
    # 99+98=197, need 3 - no. 99+97=196, need 4 - no. ...
    # Three elements: 51+52+53=156, max three: 100+99+98=297>200.
    # 100+99+1? No. 100+98+2? No. 100+97+3? No... 100+51+49? No.
    # 99+98+3? No. 99+97+4? No... 99+51+50? No. 98+97+5? No...
    # Actually 100+50+50? No, distinct. 100+49+51? 49 not in set.
    # 99+51+50? 50 not in set. 98+52+50? No.
    # It seems 51..100 might be safe! Let's verify: any subset of {51,...,100} summing to 200?
    # Min element 51. If we use k elements, sum >= 51*k. For k=1: max 100<200. k=2: min 102, max 199 (100+99). 
    # So 2-element subsets: max is 199, so no 2-element subset sums to 200.
    # k=3: min 51+52+53=156, max 100+99+98=297. Can three sum to 200?
    # 100+99+1? No. 100+98+2? No. 100+97+3? No... 100+a+b=200 => a+b=100, a,b in [51,99].
    # a+b=100 with a,b>=51: (51,49) no, (52,48) no... min a=51, then b=49<51. So no 3-element with 100.
    # 99+a+b=200 => a+b=101, a,b in [51,98]. (51,50) no, (52,49) no... min a=51,b=50<51. No.
    # 98+a+b=200 => a+b=102, a,b in [51,97]. (51,51) - but distinct! (52,50) no. Min distinct: 51+52=103>102. No.
    # So no 3-element subset. k=4: min 51+52+53+54=210>200. So no subset of {51..100} sums to 200!
    # That's 50 elements.

    # Can we do better? The greedy seed might already give us close to 50.
    # Let's see what greedy gives us.

    # Actually, let's try to improve: can we add any numbers from 1..50 to {51..100}?
    # If we add x in 1..50, we need to check if any subset of {51..100} plus x sums to 200.
    # i.e., is there a subset of {51..100} summing to 200-x?
    # For x=1: need subset of {51..100} summing to 199. 100+99=199! So can't add 1.
    # For x=2: need 198. 100+98=198. Can't add 2.
    # ...
    # For x in 1..49: need 200-x in [151,199]. 100+(100-x) = 200-x if 100-x in [51,100] i.e. x in [0,49].
    # So for x=1..49, 100+(100-x) is a valid pair in {51..100} (since 100-x in [51,99]).
    # So we can't add any of 1..49.
    # For x=50: need 150. Subset of {51..100} summing to 150? 51+99=150, 52+98=150, etc. Yes. Can't add 50.
    # So {51..100} is maximal at 50.

    # But maybe a different 50+ element set exists? 
    # Total elements is 100. We're excluding 50. Can we exclude fewer?
    # The constraint is: no subset sums to 200. 
    # This is equivalent to: the set is "200-sum-free".
    # 
    # Let's try a different approach: what if we take all of 1..100 except a small set?
    # That's hard. Let's just use the local search result.

    # Verify our current best
    result = selected
    # Ensure it's valid
    r = 1
    for x in result:
        r |= (r << x) & mask
    if (r >> TARGET) & 1:
        # Fallback to {51..100}
        result = list(range(51, 101))
    
    return result
# EVOLVE-BLOCK-END
