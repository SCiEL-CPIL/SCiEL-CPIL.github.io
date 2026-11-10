"""Combined: fast validation + 1-out/2-in and 2-out/2-in SA moves."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = 85.0

    def is_valid(seq):
        """Check if all consecutive block products are distinct. Fast early termination."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def all_products(seq):
        """Return set of all consecutive block products."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def greedy_init():
        """Standard greedy: iterate 2..128, add if no tail-product collision."""
        values = []
        products = set()
        for x in range(2, 129):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    def greedy_randomized(rng, seed_val=2):
        """Randomized greedy: try values in random order, keep valid ones."""
        values = []
        products = set()
        candidates = list(range(seed_val, 129))
        rng.shuffle(candidates)
        for x in candidates:
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        values.sort()
        return values

    def try_insert(seq, x, pos, prods):
        """Try inserting x at pos into seq. Returns (new_seq, new_prods) if valid, else (None, None)."""
        n = len(seq)
        if pos > 0 and seq[pos-1] >= x:
            return None, None
        if pos < n and x >= seq[pos]:
            return None, None
        
        # Compute new tail products that start at pos and extend right
        # Also need to check products ending at pos-1 that now extend through x
        # Actually, we need to check ALL new products involving x
        # New products are: 
        #   - products [i..j] where i <= pos-1 and j >= pos (these now include x)
        #   - singleton [x]
        # Simpler: compute all products in new_seq and check against old prods minus affected ones
        # But that's O(n^2). Let's be smarter.
        
        # New products that include x at position pos:
        # For each left start i in [0, pos], right end j in [pos, n-1] in new_seq
        # In old seq, left start i corresponds to old index i (if i < pos) 
        # and right end j corresponds to old index j-1 (if j > pos)
        
        # Products NOT involving x remain the same.
        # Products involving x: for i from 0 to pos (in new_seq), j from pos to n (in new_seq)
        #   new_seq[i..j] includes x at position pos
        #   left part: new_seq[i..pos-1] = seq[i..pos-1] if i < pos, else empty
        #   right part: new_seq[pos+1..j] = seq[pos..j-1] if j > pos, else empty
        
        new_prods = set(prods)
        
        # Remove products that are "broken" by insertion - actually no products are removed,
        # only new ones are added. Wait, no: when we insert, old products that span the
        # insertion point are no longer valid as-is. But actually they DO still exist in the
        # new sequence if we consider them... no wait.
        #
        # In old seq, product seq[i..j] for i < pos <= j doesn't exist as a contiguous block
        # in the same form. Actually it DOES - seq[i..j] is still contiguous in new_seq as
        # new_seq[i..j+1] which includes x. So old products that span pos are REPLACED.
        #
        # Let me think differently. Just compute all new products and check.
        # For efficiency, only check products that involve x.
        
        # Products involving x:
        # Left side: multiply seq[i..pos-1] for i = 0..pos
        # Right side: multiply seq[pos..j] for j = pos..n-1
        # Combined: left_prod * x * right_prod
        
        left_prods = [1]  # left_prod[i] = product of seq[i..pos-1]
        for i in range(pos - 1, -1, -1):
            left_prods.append(left_prods[-1] * seq[i])
        # left_prods[k] = product of seq[pos-k..pos-1] for k=0..pos
        # i.e., left_prods[0]=1, left_prods[1]=seq[pos-1], left_prods[2]=seq[pos-2]*seq[pos-1], etc.
        
        right_prods = [1]  # right_prod[j] = product of seq[pos..j] for j=pos..n-1
        rp = 1
        for j in range(pos, n):
            rp *= seq[j]
            right_prods.append(rp)
        # right_prods[k] = product of seq[pos..pos+k-1] for k=0..n-pos
        
        for lp in left_prods:
            for rp2 in right_prods:
                p = lp * x * rp2
                if p in new_prods:
                    return None, None
                new_prods.add(p)
        
        new_seq = seq[:pos] + [x] + seq[pos:]
        return new_seq, new_prods

    def try_remove(seq, pos, prods):
        """Remove element at pos from seq. Returns (new_seq, new_prods) or (None, None)."""
        n = len(seq)
        x = seq[pos]
        new_seq = seq[:pos] + seq[pos+1:]
        
        # Products that involved x are removed. All other products remain.
        # New products that are formed: none (removing just breaks blocks)
        # Wait - removing x means that blocks that spanned x are now split.
        # So old products involving x are gone, and no new products are formed.
        # Actually that's wrong too. Consider seq = [a, x, b]. Removing x gives [a, b].
        # Old products: a, x, b, a*x, x*b, a*x*b
        # New products: a, b, a*b
        # So a*b is NEW, and a*x, x*b, a*x*b are REMOVED.
        #
        # So we need to: remove all products involving x, and add new products that 
        # connect the left and right parts.
        
        # Products involving x in old seq:
        # For i in [0, pos], j in [pos, n-1]: product seq[i..j]
        
        left_prods = [1]
        for i in range(pos - 1, -1, -1):
            left_prods.append(left_prods[-1] * seq[i])
        
        right_prods = [1]
        rp = 1
        for j in range(pos, n):
            rp *= seq[j]
            right_prods.append(rp)
        
        # Remove products involving x
        new_prods = set()
        for p in prods:
            # Check if p involves x - this is hard to do directly
            # Better: rebuild prods from scratch for the new seq
            pass
        
        # Simpler: just rebuild all products for new_seq
        # But that's O(n^2). For now, let's just do it.
        new_prods = all_products(new_seq)
        
        return new_seq, new_prods

    def local_search_insertions(seq, max_iterations=500, deadline=85.0):
        """Try to improve by inserting missing values at valid positions."""
        seq = seq[:]
        in_seq = set(seq)
        missing = [x for x in range(2, 129) if x not in in_seq]
        prods = all_products(seq)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline - 5:
                break
            random.shuffle(missing)
            for x in missing:
                if time.time() > deadline - 5:
                    break
                n = len(seq)
                found = False
                for pos in range(n + 1):
                    new_seq, new_prods = try_insert(seq, x, pos, prods)
                    if new_seq is not None:
                        seq = new_seq
                        prods = new_prods
                        in_seq.add(x)
                        missing.remove(x)
                        improved = True
                        found = True
                        break
                if found:
                    break
        return seq

    def get_neighbor_fast(seq, prods, rng, candidates):
        """Generate a valid neighbor using incremental product tracking."""
        n = len(seq)
        if n == 0 or not candidates:
            return None, None

        move_type = rng.random()

        if move_type < 0.35:
            # 1-out/1-in
            rm_pos = rng.randint(0, n - 1)
            x = rng.choice(candidates)
            # Remove first, then insert
            new_seq1 = seq[:rm_pos] + seq[rm_pos + 1:]
            # Check monotonicity after removal
            ok = True
            for i in range(len(new_seq1) - 1):
                if new_seq1[i] >= new_seq1[i + 1]:
                    ok = False
                    break
            if not ok:
                return None, None
            # Now insert x into new_seq1
            base_n = len(new_seq1)
            pos = rng.randint(0, base_n)
            new_seq, new_prods = try_insert(new_seq1, x, pos, all_products(new_seq1))
            if new_seq is None:
                return None, None
            return new_seq, new_prods

        elif move_type < 0.65:
            # 1-out/2-in
            rm_pos = rng.randint(0, n - 1)
            if len(candidates) < 2:
                x = rng.choice(candidates)
                new_seq1 = seq[:rm_pos] + seq[rm_pos + 1:]
                ok = True
                for i in range(len(new_seq1) - 1):
                    if new_seq1[i] >= new_seq1[i + 1]:
                        ok = False
                        break
                if not ok:
                    return None, None
                base_n = len(new_seq1)
                pos = rng.randint(0, base_n)
                new_seq, new_prods = try_insert(new_seq1, x, pos, all_products(new_seq1))
                if new_seq is None:
                    return None, None
                return new_seq, new_prods
            x1, x2 = rng.sample(candidates, 2)
            new_seq1 = seq[:rm_pos] + seq[rm_pos + 1:]
            ok = True
            for i in range(len(new_seq1) - 1):
                if new_seq1[i] >= new_seq1[i + 1]:
                    ok = False
                    break
            if not ok:
                return None, None
            base_n = len(new_seq1)
            pos1 = rng.randint(0, base_n)
            pos2 = rng.randint(0, base_n + 1)
            if pos2 > pos1:
                pos2 += 1
            # Insert x1 at pos1, then x2 at pos2
            new_seq_a, prods_a = try_insert(new_seq1, x1, pos1, all_products(new_seq1))
            if new_seq_a is None:
                return None, None
            # Adjust pos2 for the insertion of x1
            if pos2 > pos1:
                pos2_adj = pos2
            else:
                pos2_adj = pos2
            new_seq, new_prods = try_insert(new_seq_a, x2, pos2_adj, prods_a)
            if new_seq is None:
                return None, None
            return new_seq, new_prods

        elif move_type < 0.85:
            # 2-out/2-in
            if n < 2:
                return None, None
            rm1, rm2 = rng.sample(range(n), 2)
            if rm1 > rm2:
                rm1, rm2 = rm2, rm1
            # Remove both
            new_seq1 = seq[:rm2] + seq[rm2 + 1:rm1] + seq[rm1 + 1:]
            ok = True
            for i in range(len(new_seq1) - 1):
                if new_seq1[i] >= new_seq1[i + 1]:
                    ok = False
                    break
            if not ok:
                return None, None
            base_n = len(new_seq1)
            if len(candidates) < 2:
                x = rng.choice(candidates)
                pos = rng.randint(0, base_n)
                new_seq, new_prods = try_insert(new_seq1, x, pos, all_products(new_seq1))
                if new_seq is None:
                    return None, None
                return new_seq, new_prods
            x1, x2 = rng.sample(candidates, 2)
            pos1 = rng.randint(0, base_n)
            pos2 = rng.randint(0, base_n + 1)
            if pos2 > pos1:
                pos2 += 1
            new_seq_a, prods_a = try_insert(new_seq1, x1, pos1, all_products(new_seq1))
            if new_seq_a is None:
                return None, None
            if pos2 > pos1:
                pos2_adj = pos2
            else:
                pos2_adj = pos2
            new_seq, new_prods = try_insert(new_seq_a, x2, pos2_adj, prods_a)
            if new_seq is None:
                return None, None
            return new_seq, new_prods

        else:
            # Swap two elements (only if result is still increasing)
            if n < 2:
                return None, None
            i, j = rng.sample(range(n), 2)
            if i > j:
                i, j = j, i
            if seq[i] == seq[j]:
                return None, None
            # After swap, check if still increasing
            new_seq = seq[:]
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            ok = True
            for k in range(len(new_seq) - 1):
                if new_seq[k] >= new_seq[k + 1]:
                    ok = False
                    break
            if not ok:
                return None, None
            new_prods = all_products(new_seq)
            return new_seq, new_prods

    # Strategy 1: Multiple greedy initializations
    best = greedy_init()
    best_len = len(best)

    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 40:
            break
        rng_g = random.Random(seed)
        cand = greedy_randomized(rng_g)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)

    # Strategy 2: Targeted insertion search
    current = best[:]
    improved_seq = local_search_insertions(current, max_iterations=200, deadline=deadline)
    if len(improved_seq) > best_len:
        best = improved_seq
        best_len = len(best)

    # Strategy 3: Simulated annealing with incremental products
    rng_sa = random.Random(42)
    cur = best[:]
    cur_len = len(cur)
    cur_prods = all_products(cur)
    best_sa = cur[:]
    best_sa_len = cur_len
    in_seq = set(cur)
    candidates = [x for x in range(2, 129) if x not in in_seq]
    T = 1.0
    T_min = 0.01
    cooling = 0.99995
    no_improve = 0
    sa_steps = 0

    while time.time() < deadline - 5 and T > T_min:
        neighbor, neighbor_prods = get_neighbor_fast(cur, cur_prods, rng_sa, candidates)
        sa_steps += 1
        if neighbor is not None:
            delta_len = len(neighbor) - cur_len
            if delta_len >= 0:
                cur = neighbor
                cur_len = len(cur)
                cur_prods = neighbor_prods
                in_seq = set(cur)
                candidates = [x for x in range(2, 129) if x not in in_seq]
                if cur_len > best_sa_len:
                    best_sa = cur[:]
                    best_sa_len = cur_len
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if rng_sa.random() < (2.718281828 ** (delta_len / T)):
                    cur = neighbor
                    cur_len = len(cur)
                    cur_prods = neighbor_prods
                    in_seq = set(cur)
                    candidates = [x for x in range(2, 129) if x not in in_seq]
                no_improve += 1
        else:
            no_improve += 1

        T *= cooling
        if no_improve > 5000:
            cur = best_sa[:]
            cur_len = best_sa_len
            cur_prods = all_products(cur)
            in_seq = set(cur)
            candidates = [x for x in range(2, 129) if x not in in_seq]
            no_improve = 0
            T = max(T, 0.5)

    if best_sa_len > best_len:
        best = best_sa
        best_len = best_sa_len

    # Strategy 4: Final targeted insertion pass
    final_seq = local_search_insertions(best, max_iterations=100, deadline=deadline)
    if len(final_seq) > best_len:
        best = final_seq
        best_len = len(best)

    # Final verification
    if not is_valid(best):
        best = greedy_init()

    return best

# EVOLVE-BLOCK-END
