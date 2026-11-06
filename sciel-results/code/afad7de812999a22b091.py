import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    n = 192
    max_sum = 3 * n  # 576

    # Precompute divisors for all sums up to max_sum
    # divisors[s] = list of d in [1, n] such that d | s
    divisors = [[] for _ in range(max_sum + 1)]
    for d in range(1, n + 1):
        for s in range(d, max_sum + 1, d):
            divisors[s].append(d)

    # Precompute all valid triples (x, y, z) with 1 <= x < y < z <= n
    # and their sum.
    # We will use this to quickly check conflicts.
    # Actually, for the "can_add" logic, we need to know if a triple in S sums to a multiple of a.
    # And if a triple involving a sums to a multiple of b in S.
    
    # Let's stick to the O(K^2) check but optimize the inner loop.
    
    def can_add(a, S):
        """
        Check if adding 'a' to set S results in a valid set.
        Returns True if valid.
        """
        S_list = list(S)
        S_set = set(S)
        K = len(S_list)
        
        # 1. Check if 'a' divides the sum of any 3 distinct elements in S.
        # We need to find x, y, z in S such that (x+y+z) % a == 0.
        # x, y, z are distinct.
        
        # Optimization: 
        # For each pair (x, y) in S, let rem = (-x-y) % a.
        # We need a third element z in S, z != x, z != y, such that z % a == rem.
        
        # We can maintain a count of residues in S modulo a?
        # But 'a' changes, so we can't precompute for all 'a' easily without heavy memory.
        # However, K is small (~48).
        # Let's iterate pairs (x, y).
        # There are K*(K-1)/2 pairs.
        # For each pair, we check if there is a z.
        
        # To check for z efficiently:
        # We can precompute a dictionary: residue -> count of elements in S with that residue mod a.
        # And also residue -> list of elements? Or just count is enough if we handle distinctness carefully.
        
        # Distinctness:
        # If we pick x, y, we need z != x and z != y.
        # If residue rem has count c in S:
        #   - If x % a == rem, then x is a candidate. If y % a == rem, y is a candidate.
        #   - We need at least one z that is NOT x and NOT y.
        #   - So, count of valid z = c - (1 if x%a==rem else 0) - (1 if y%a==rem else 0).
        #   - If this count > 0, then conflict exists.
        
        # Let's build the residue counts for 'a'.
        # This takes O(K) time.
        # Then for each pair (x, y), O(1) check.
        # Total O(K^2) per can_add call.
        
        res_counts = {}
        for x in S_list:
            r = x % a
            res_counts[r] = res_counts.get(r, 0) + 1
        
        for i in range(K):
            x = S_list[i]
            rx = x % a
            for j in range(i + 1, K):
                y = S_list[j]
                ry = y % a
                target_rem = (-x - y) % a
                # Note: (-x-y) % a is equivalent to (a - (x+y)%a) % a
                # Python's % handles negative numbers correctly (returns non-negative result for positive modulus)
                
                count = res_counts.get(target_rem, 0)
                # Subtract x if it matches
                if rx == target_rem:
                    count -= 1
                # Subtract y if it matches
                if ry == target_rem:
                    count -= 1
                
                if count > 0:
                    return False
        
        # 2. Check if any b in S divides the sum of any triple involving 'a'.
        # Triples are (a, u, v) where u, v in S, u != v.
        # Sum T = a + u + v.
        # We need to check if any b in S (b != a, b != u, b != v) divides T.
        
        # Iterate over pairs (u, v) in S.
        # For each pair, T = a + u + v.
        # Check divisors of T.
        # If any divisor b is in S and b is not a, u, or v, return False.
        
        for i in range(K):
            u = S_list[i]
            for j in range(i + 1, K):
                v = S_list[j]
                T = a + u + v
                if T <= max_sum:
                    for b in divisors[T]:
                        if b == a or b == u or b == v:
                            continue
                        if b in S_set:
                            return False
        
        return True

    # Greedy initialization
    # Try different orders to see which yields a better starting point.
    # Largest first is usually good for packing.
    
    best_S = set()
    best_size = 0
    
    # Multi-start greedy
    orders = [
        list(range(n, 0, -1)),  # Descending
        list(range(1, n + 1)),  # Ascending
        list(range(n, 0, -1, 2)), # Descending even
        list(range(n, 0, -1, 2)) + list(range(n-1, 0, -1, 2)), # Descending even then odd
    ]
    
    # Add some random shuffles
    for _ in range(5):
        random_order = list(range(1, n + 1))
        random.shuffle(random_order)
        orders.append(random_order)

    for order in orders:
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        if time.time() - start_time > time_limit * 0.2:
            break

    # Local Search
    # We will use the best_S found so far.
    S = set(best_S)
    
    # To speed up local search, we can cache the "conflict" status?
    # Or just use can_add which is O(K^2 + K^2 * D) where D is avg divisors.
    # D is small (max divisors for 576 is ~30, avg is much less).
    # So can_add is roughly O(K^2). K=48, K^2=2304.
    # 2304 ops per check.
    # In 50 seconds, we can do millions of checks.
    
    while time.time() - start_time < time_limit:
        S_list = list(S)
        U_not_S = [x for x in range(1, n + 1) if x not in S]
        
        if not U_not_S:
            break
        
        # Try Add
        if random.random() < 0.3:
            a = random.choice(U_not_S)
            if can_add(a, S):
                S.add(a)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        else:
            # Try Swap
            # Pick a random element to remove and a random element to add
            r = random.choice(S_list)
            a = random.choice(U_not_S)
            
            # Check if S - {r} + {a} is valid
            # This is equivalent to checking if 'a' can be added to S - {r}
            # AND that removing 'r' doesn't break anything (it can't, removing elements preserves validity of remaining subsets)
            # Wait, is that true?
            # If S is valid, is S - {r} valid? Yes.
            # So we just need to check if 'a' is compatible with S - {r}.
            # The function can_add(a, S_mid) does exactly this.
            
            S_mid = S - {r}
            if can_add(a, S_mid):
                S.remove(r)
                S.add(a)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                # Accept lateral moves to explore
                # But to avoid cycling, we might want to be careful.
                # Simple hill climbing with random restarts or simulated annealing?
                # For now, always accept valid swaps.
                
                # Occasionally, try to remove an element to allow adding two?
                # That's a 2-exchange. More complex.
                # Let's stick to 1-exchange for now.
                
        # Occasionally, try a "kick" or restart from a different greedy solution?
        # If stuck for a while, we might want to diversify.
        # For simplicity, let's just keep going.
        
        # Safety check: if we've been running for a long time without improvement,
        # we might try to force a restart from the best found so far with a different random seed?
        # No, the current S is a valid state. We are just exploring its neighborhood.
        
    # Return the best set found
    return list(best_S)

# Quick self-test to ensure no syntax errors or immediate crashes
if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} elements")
    print(res)
