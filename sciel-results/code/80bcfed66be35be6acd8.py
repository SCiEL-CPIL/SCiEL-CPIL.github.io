import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # 1. Generate a maximal Sidon set M (the "universe" of potential elements)
    # We use a greedy approach to build a maximal Sidon set.
    # To ensure it's "maximal" (inclusion-wise), we iterate through all numbers.
    # A number is added if it doesn't create a collision.
    # If it does create a collision, it's "blocked" by the current set.
    # At the end, no number can be added, so it is maximal.
    
    def generate_maximal_sidon():
        selected = []
        sums = set()
        for x in range(1, N + 1):
            # Check if x can be added
            # New sums: 2x, x+y for y in selected
            # Collision if any new sum is in sums, or if new sums collide with each other?
            # New sums are 2x and x+y.
            # 2x == x+y => x=y (impossible since x not in selected)
            # x+y1 == x+y2 => y1=y2 (impossible)
            # So internal collision of new sums is impossible.
            # Only check against existing sums.
            
            added = [2 * x] + [x + y for y in selected]
            if any(s in sums for s in added):
                continue
            
            selected.append(x)
            sums.update(added)
        return selected, sums

    M, M_sums = generate_maximal_sidon()
    
    # M is a maximal Sidon set.
    # We want to find a subset S of M such that S is also maximal.
    # S is maximal if and only if for every x in [1, N] \ S, x is "blocked" by S.
    # Since M is maximal, any x not in M is already blocked by M.
    # So we only need to worry about x in M \ S.
    # If x in M \ S, is x blocked by S?
    # x is blocked by S if adding x to S creates a collision.
    # The collision would be with some sums in S.
    # Let's define for each x in M, the set of elements in M that "block" x.
    # Actually, let's look at it this way:
    # For each x in M, let B(x) be the set of y in M such that {x, y} is NOT a Sidon set?
    # No, that's not right.
    #
    # Let's re-evaluate.
    # We want S subset M such that S is maximal.
    # Condition: For all z in [1, N] \ S, z is not addable to S.
    # Case 1: z not in M. Since M is maximal, z is not addable to M.
    #          Since S subset M, is z addable to S?
    #          If z is addable to S, then z + s1 != z + s2 (trivial) and z+s != 2s' etc.
    #          The condition for z being addable to S is that no sum in S equals 2z or z+s for s in S.
    #          If z is NOT addable to M, it means there exists s1, s2 in M (possibly s1=s2)
    #          such that 2z == s1+s2 OR z+s1 == s1+s2 (impossible) ...
    #          Actually, z is not addable to M if there exist a, b in M (a<=b) such that a+b = 2z or a+b = z+c?
    #          No, z is not addable to M if the set of sums of M intersects {2z} U {z+m | m in M}.
    #          Let S_M(z) = {2z} U {z+m | m in M}.
    #          z is blocked by M if S_M(z) intersects Sums(M).
    #
    #          Now consider S subset M.
    #          z is blocked by S if S_M(z) intersects Sums(S).
    #          Since Sums(S) is a subset of Sums(M), if z is blocked by M, is it necessarily blocked by S?
    #          NO. The collision in M might involve elements not in S.
    #          Example: M = {1, 2, 4, 8}. z=3.
    #          Sums(M) = {2, 3, 5, 4, 6, 8, 10, 16}.
    #          S_M(3) = {6, 3+1=4, 3+2=5, 3+4=7, 3+8=11}.
    #          Intersection: {4, 5, 6}. So 3 is blocked by M.
    #          If S = {1, 8}, Sums(S) = {2, 9, 16}.
    #          Intersection of S_M(3) and Sums(S) is empty.
    #          So 3 is NOT blocked by S. S is not maximal.
    #
    # So we cannot just pick any subset. We must ensure that for every z in [1, N] \ S,
    # the "witness" of the collision in M is preserved in S.
    #
    # Let's define for each z in [1, N] \ M (which are always blocked by M, and we must keep them blocked by S):
    # Let W_M(z) be the set of pairs (a, b) in M x M (a<=b) such that a+b is in S_M(z).
    # For S to block z, S must contain at least one element from the "witness" sets?
    # Actually, if a+b = 2z, then a and b must be in S? No, just the sum a+b must be in Sums(S).
    # This requires a in S and b in S.
    # If a+b = z+c (where c in M), then we need a in S, b in S, and c in S?
    # Wait. The sum a+b is in Sums(S) iff a in S and b in S.
    # The sum z+c is in S_M(z).
    # If a+b = z+c, and a,b,c in S, then a+b is in Sums(S) and z+c is in S_M(z).
    # So the collision holds.
    #
    # So for each z, we have a set of "clauses" (witnesses).
    # A witness is a set of elements in M that must be present in S to create the collision.
    # If the collision is 2z = a+b, the witness is {a, b}.
    # If the collision is a+b = z+c, the witness is {a, b, c}.
    #
    # We need to select S subset M such that for every z in [1, N] \ S, at least one witness for z is fully contained in S.
    # Note: If z is in S, we don't need to block z.
    # So the constraint is: For all z in [1, N] \ S, S hits at least one witness set W(z).
    #
    # This is a Hitting Set problem (or Set Cover dual).
    # Universe: M.
    # Sets: For each z not in S, we need to satisfy the condition.
    # But S determines which z are excluded.
    #
    # Let's try a greedy approach for S:
    # 1. Start with S = M.
    # 2. Try to remove elements from S one by one.
    # 3. When removing x, check if S \ {x} is still maximal.
    #    This requires checking all z in [1, N] \ (S \ {x}) to see if they are blocked.
    #    This is expensive if done naively.
    #
    # Alternative:
    # Start with S = empty? No, must be non-empty.
    #
    # Let's use the Hitting Set formulation more directly.
    # We want to minimize |S| subject to:
    # For all z in [1, N] \ S, S intersects (or contains) a witness for z.
    #
    # This is tricky because the set of z depends on S.
    # However, if we assume S is small, most z are in [1, N] \ S.
    #
    # Let's try a different angle.
    # The "essential" elements of M are those that cannot be removed without breaking maximality.
    # Let E be the set of essential elements.
    # Any maximal subset S of M must contain E?
    # Not necessarily. If we remove an essential element x, we might add a non-essential element y to compensate?
    # But we are restricted to subsets of M?
    # The task says "Construct a Sidon subset...". It doesn't say it must be a subset of a specific maximal set.
    # But if we find a maximal set M, any other maximal set S might share elements or not.
    # However, it is known that maximal Sidon sets are "close" to each other?
    #
    # Let's stick to the "remove from M" strategy.
    # 1. Generate M.
    # 2. Identify "critical" elements.
    #    For each x in M, check if M \ {x} is maximal.
    #    If M \ {x} is NOT maximal, then x is "essential" in the sense that removing it breaks maximality.
    #    Let E be the set of such x.
    #    If we pick S = E, is S maximal?
    #    Not necessarily. Removing x might make some z unblocked.
    #    But if we keep all essential elements, do we block everything?
    #    If M \ {x} is not maximal, there exists z such that z is not blocked by M \ {x}.
    #    This z was blocked by M. The block was witnessed by x.
    #    So if we keep x, z is blocked.
    #    So if S = E, then for any z, if z was blocked by M via a witness involving only non-essential elements, S might fail.
    #    But if a witness involves an essential element, and we keep all essential elements, is it enough?
    #
    # Let's test this hypothesis: S = E (the set of elements whose removal breaks maximality of M).
    # Is S maximal?
    # Suppose S is not maximal. Then there is z not in S that is not blocked by S.
    # Since M is maximal, z is blocked by M.
    # So there is a witness W in M for z.
    # If W is contained in S, then z is blocked by S. Contradiction.
    # So W is NOT contained in S.
    # This means W contains some element x not in S.
    # x not in S means x is not essential.
    # So M \ {x} is maximal.
    # This implies that even without x, z is still blocked by M \ {x}.
    # So there is another witness W' in M \ {x} for z.
    # If W' is contained in S, z is blocked.
    # If W' is not contained in S, it contains y not in S (non-essential).
    #
    # It seems possible that S = E is maximal.
    # Let's try to construct S by starting with M and removing non-essential elements.
    #
    # Algorithm:
    # 1. Generate M.
    # 2. Compute E = {x in M | M \ {x} is not maximal}.
    # 3. Start with S = M.
    # 4. For each x in M \ E (non-essential), try to remove x from S.
    #    Check if S \ {x} is maximal.
    #    If yes, remove it.
    #    If no, keep it.
    # 5. Return S.
    #
    # This is a greedy removal. It might get stuck in a local minimum.
    # To improve, we can try multiple random orders of removal.
    #
    # Optimization:
    # Checking "is maximal" is O(N * |S|).
    # |S| is roughly sqrt(N) ~ 16.
    # N = 256.
    # Cost per check ~ 4000 ops.
    # Removals ~ 16.
    # Total ~ 64k ops per run.
    # We can do many runs in 55s.

    def is_maximal(selected, sums):
        selected_set = set(selected)
        for z in range(1, N + 1):
            if z in selected_set:
                continue
            # Check if z can be added
            # Sums to check: 2z, z+y for y in selected
            blocked = False
            if 2 * z in sums:
                blocked = True
            else:
                for y in selected:
                    if z + y in sums:
                        blocked = True
                        break
            if not blocked:
                return False
        return True

    best_set = None
    best_size = float('inf')

    # We will run multiple iterations of the greedy removal on M.
    # To diversify, we can shuffle the order of non-essential elements.
    # Also, we can try to generate different M's?
    # The greedy generation of M is deterministic if we iterate 1..N.
    # We can randomize the order of checking 1..N to get different maximal sets M.
    
    num_restarts = 1000
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Generate a random maximal Sidon set
        # Shuffle the range
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        M = []
        M_sums = set()
        for x in candidates:
            added = [2 * x] + [x + y for y in M]
            if any(s in M_sums for s in added):
                continue
            M.append(x)
            M_sums.update(added)
            
        # M is maximal by construction (we checked all candidates)
        
        # Identify essential elements
        # x is essential if M \ {x} is not maximal
        essential = set()
        for x in M:
            # Check if M \ {x} is maximal
            # Optimization: We don't need to recompute sums from scratch?
            # M_sums is the sums of M.
            # Sums of M \ {x} are M_sums minus sums involving x?
            # No, sums are a+b. If we remove x, we lose 2x and x+y.
            # But other sums remain.
            # We need the set of sums of M \ {x}.
            # Let's just compute is_maximal for M \ {x}.
            # To speed up, we can pass the sums.
            # But removing x changes the sums.
            # We can compute new_sums = M_sums - {2x, x+y for y in M, y!=x}
            # Wait, if a+b = x+y, and we remove x, is a+b still a sum?
            # Yes, if a,b != x.
            # So we must be careful.
            # It's safer to just call is_maximal with the new set and recompute sums?
            # Or better: is_maximal can take the set and compute sums internally?
            # That's O(|S|^2). |S|~16. 256 ops.
            # is_maximal is O(N * |S|) = 4096 ops.
            # Total per check ~ 4300 ops.
            # 16 checks ~ 70k ops.
            # 1000 restarts ~ 70M ops. Might be too slow in Python.
            #
            # Let's optimize is_maximal.
            # We can precompute the "blocking" for each z?
            
            # Let's try to be smarter.
            # Instead of checking all x in M, we can just try to remove them greedily.
            # Start with S = M.
            # Iterate through elements in random order.
            # Try to remove.
            
            pass # Placeholder, we will do the greedy removal directly

        # Greedy Removal
        S = list(M)
        S_sums = set(M_sums)
        S_set = set(S)
        
        # Shuffle S to randomize removal order
        random.shuffle(S)
        
        # We iterate through S and try to remove each element
        # We must be careful: removing one element might make another element "removable" or "unremovable"?
        # Actually, if we remove x, the set gets smaller.
        # Maximality is a property of the set.
        # If S is maximal, is S \ {x} maximal?
        # Not necessarily.
        # If S \ {x} is maximal, we remove x.
        # If not, we keep x.
        #
        # Note: If we keep x, it might block the removal of y?
        # No, removal of y is checked against the current S.
        #
        # Let's do a single pass.
        # To improve, we can do multiple passes until no more removals are possible.
        
        changed = True
        while changed:
            changed = False
            # Shuffle to break symmetry
            random.shuffle(S)
            for x in S:
                if time.time() > deadline:
                    break
                # Try removing x
                new_S = [y for y in S if y != x]
                if not new_S:
                    continue # Must be non-empty
                # Check if new_S is maximal
                # Compute sums of new_S
                new_sums = set()
                for i in range(len(new_S)):
                    a = new_S[i]
                    new_sums.add(2 * a)
                    for j in range(i + 1, len(new_S)):
                        new_sums.add(a + new_S[j])
                
                if is_maximal(new_S, new_sums):
                    S = new_S
                    S_set = set(S)
                    S_sums = new_sums
                    changed = True
            if time.time() > deadline:
                break

        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)

    if best_set is None:
        # Fallback: return the first maximal set generated
        M, _ = generate_maximal_sidon()
        return M

    return best_set
# EVOLVE-BLOCK-END
