import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    def calc_score(mask):
        """Count distinct d > 0 where a, a+d, a+2d all in set."""
        s = 0
        m = mask
        for d in range(1, 257):
            if m & (m >> d) & (m >> (2 * d)):
                s += 1
        return s

    def get_elements(mask):
        elems = []
        for i in range(1, 513):
            if mask & (1 << i):
                elems.append(i)
        return elems

    def mask_from_elems(elems):
        mask = 0
        for e in elems:
            mask |= (1 << e)
        return mask

    def random_mask():
        elems = random.sample(range(1, 513), 32)
        return mask_from_elems(elems)

    def structured_mask(type_id):
        mask = 0
        if type_id == 0:
            # Contiguous block
            for i in range(1, 33):
                mask |= (1 << i)
        elif type_id == 1:
            # Mix of small cluster and spread elements
            # 16 elements in [1,32], 16 spread across [1,512]
            cluster = random.sample(range(1, 33), 16)
            spread = random.sample(range(50, 513), 16)
            for e in cluster + spread:
                mask |= (1 << e)
        elif type_id == 2:
            # Two clusters at different scales
            # 16 in [1,40], 16 in [200,512]
            c1 = random.sample(range(1, 41), 16)
            c2 = random.sample(range(200, 513), 16)
            for e in c1 + c2:
                mask |= (1 << e)
        elif type_id == 3:
            # Geometric progression based
            vals = set()
            base = 1
            while len(vals) < 32 and base <= 512:
                vals.add(base)
                base = int(base * 1.15) + 1
            if len(vals) < 32:
                pool = [x for x in range(1, 513) if x not in vals]
                vals.update(random.sample(pool, 32 - len(vals)))
            for v in vals:
                mask |= (1 << v)
        elif type_id == 4:
            # Three clusters
            for start in [1, 150, 350]:
                for i in range(11):
                    mask |= (1 << (start + i))
            # Add 1 more
            mask |= (1 << 500)
        elif type_id == 5:
            # Dense small + medium + sparse large
            for i in range(1, 17):
                mask |= (1 << i)
            for i in range(16, 32):
                mask |= (1 << (i * 10))
            for i in range(32, 48):
                mask |= (1 << (i * 10 + 5))
        elif type_id == 6:
            # Every 16th number
            for i in range(32):
                mask |= (1 << (1 + i * 16))
        elif type_id == 7:
            # Hybrid: 8 dense, 8 medium, 8 sparse, 8 very sparse
            for i in range(8):
                mask |= (1 << (1 + i))
            for i in range(8):
                mask |= (1 << (20 + i * 5))
            for i in range(8):
                mask |= (1 << (100 + i * 20))
            for i in range(8):
                mask |= (1 << (300 + i * 25))
        return mask

    best_mask = 0
    best_score = -1

    # Generate diverse initializations
    inits = []
    for t in range(8):
        inits.append(structured_mask(t))
    for _ in range(6):
        inits.append(random_mask())

    # Evaluate initializations
    for m in inits:
        s = calc_score(m)
        if s > best_score:
            best_score = s
            best_mask = m

    # Simulated Annealing with targeted moves
    time_per_restart = (deadline - start_time - 5) / len(inits)

    for r in range(len(inits)):
        if time.time() > deadline - 3:
            break

        current_mask = inits[r]
        current_score = calc_score(current_mask)

        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask

        T = 2.0
        cool_rate = 0.9997
        min_T = 0.005
        restart_deadline = min(start_time + (r + 1) * time_per_restart, deadline - 3)

        no_improve = 0
        max_no_improve = 3000

        while time.time() < restart_deadline - 0.5:
            if T < min_T:
                break

            # Choose move type: 1-swap (70%), 2-swap (20%), targeted (10%)
            move_type = random.random()
            
            if move_type < 0.7:
                # 1-swap: replace one element with another
                x = random.randint(1, 512)
                y = random.randint(1, 512)
                if x == y:
                    continue
                bx, by = 1 << x, 1 << y
                if (current_mask & bx) and not (current_mask & by):
                    new_mask = current_mask ^ bx ^ by
                    new_score = calc_score(new_mask)
                    delta = new_score - current_score
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        no_improve = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                    else:
                        no_improve += 1

            elif move_type < 0.9:
                # 2-swap
                x1, x2 = random.sample(range(1, 513), 2)
                y1, y2 = random.sample(range(1, 513), 2)
                if x1 in (y1, y2) or x2 in (y1, y2):
                    continue
                bx1, bx2, by1, by2 = 1 << x1, 1 << x2, 1 << y1, 1 << y2
                if (current_mask & bx1) and (current_mask & bx2) and not (current_mask & by1) and not (current_mask & by2):
                    new_mask = current_mask ^ bx1 ^ bx2 ^ by1 ^ by2
                    new_score = calc_score(new_mask)
                    delta = new_score - current_score
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        no_improve = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                    else:
                        no_improve += 1

            else:
                # Targeted move: try to create a new AP for a currently missing d
                # Find a d that's not currently achieved
                missing_ds = []
                m = current_mask
                for d in range(1, 257):
                    if not (m & (m >> d) & (m >> (2 * d))):
                        missing_ds.append(d)
                
                if missing_ds:
                    d = random.choice(missing_ds)
                    # Try to find a, a+d, a+2d where we can adjust
                    # For each possible a, check how many of {a, a+d, a+2d} are in set
                    best_a = None
                    best_missing = 3
                    for a in range(1, 513 - 2 * d + 1):
                        a1, a2, a3 = a, a + d, a + 2 * d
                        in_set = 0
                        for val in (a1, a2, a3):
                            if current_mask & (1 << val):
                                in_set += 1
                        missing = 3 - in_set
                        if missing < best_missing:
                            best_missing = missing
                            best_a = a
                        if missing == 0:
                            break
                    
                    if best_a is not None and best_missing > 0:
                        a1, a2, a3 = best_a, best_a + d, best_a + 2 * d
                        # Find elements to swap out
                        out_elems = []
                        in_elems = []
                        for val in (a1, a2, a3):
                            if not (current_mask & (1 << val)):
                                in_elems.append(val)
                        
                        if in_elems:
                            # Need to remove some elements to make room
                            # Find elements that are "least useful"
                            # Try swapping one in_elem with one out_elem
                            for in_val in in_elems:
                                # Find a candidate to remove
                                candidates = []
                                for i in range(1, 513):
                                    if current_mask & (1 << i) and i != in_val:
                                        candidates.append(i)
                                if not candidates:
                                    continue
                                out_val = random.choice(candidates)
                                bx, by = 1 << out_val, 1 << in_val
                                new_mask = current_mask ^ bx ^ by
                                new_score = calc_score(new_mask)
                                delta = new_score - current_score
                                if delta >= 0 or random.random() < math.exp(delta / T):
                                    current_mask = new_mask
                                    current_score = new_score
                                    no_improve = 0
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_mask = current_mask
                                    break

            T *= cool_rate

            if no_improve > max_no_improve:
                # Restart with perturbation
                for _ in range(30):
                    x1, x2 = random.sample(range(1, 513), 2)
                    y1, y2 = random.sample(range(1, 513), 2)
                    if x1 in (y1, y2) or x2 in (y1, y2):
                        continue
                    bx1, bx2, by1, by2 = 1 << x1, 1 << x2, 1 << y1, 1 << y2
                    if (current_mask & bx1) and (current_mask & bx2) and not (current_mask & by1) and not (current_mask & by2):
                        current_mask = current_mask ^ bx1 ^ bx2 ^ by1 ^ by2
                        current_score = calc_score(current_mask)
                        T = 1.0
                        no_improve = 0
                        break
                else:
                    break

    # Final polish: local search from best with low temperature
    if time.time() < deadline - 2 and best_mask != 0:
        current_mask = best_mask
        current_score = best_score
        T = 0.03
        cool_rate = 0.9995
        min_T = 0.001

        while time.time() < deadline - 0.5:
            if T < min_T:
                break

            x = random.randint(1, 512)
            y = random.randint(1, 512)
            if x == y:
                continue
            bx, by = 1 << x, 1 << y
            if (current_mask & bx) and not (current_mask & by):
                new_mask = current_mask ^ bx ^ by
                new_score = calc_score(new_mask)
                delta = new_score - current_score
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                T *= cool_rate

    result = get_elements(best_mask)
    if len(result) != 32:
        return list(range(1, 33))
    return result
