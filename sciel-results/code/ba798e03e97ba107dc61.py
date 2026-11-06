<<<<<<< SEARCH
        # Score function: prioritize length, then flexibility
        def calc_score(state):
            l, seq, pl, ds = state
            # Potential: count how many of the next 5 numbers are compatible
            potential = 0
            last_p = pl[-1]
            for m in range(n+1, min(n+6, 129)):
                v_m = vecs[m]
                new_p = tuple(last_p[i] + v_m[i] for i in range(num_primes))
                ok = True
                for p in pl:
                    d = tuple(new_p[i] - p[i] for i in range(num_primes))
                    if d in ds:
                        ok = False
                        break
                if ok:
                    potential += 1
            return (l, potential)
=======
        # Score function: prioritize length, then flexibility
        def calc_score(state):
            l, seq, pl, ds = state
            # Potential: count how many of the next 10 numbers are compatible
            potential = 0
            last_p = pl[-1]
            for m in range(n+1, min(n+11, 129)):
                v_m = vecs[m]
                new_p = tuple(last_p[i] + v_m[i] for i in range(num_primes))
                ok = True
                for p in pl:
                    d = tuple(new_p[i] - p[i] for i in range(num_primes))
                    if d in ds:
                        ok = False
                        break
                if ok:
                    potential += 1
            # Secondary: fewer distinct diffs = less constrained = more flexible
            return (l, potential, -len(ds))
>>>>>>> REPLACE

