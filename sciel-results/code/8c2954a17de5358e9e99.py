<<<<<<< SEARCH
    def polish(assignment, budget):
        # high-precision deterministic-ish min-conflicts polish
        sat, uc = build_sat(assignment)
        if uc == 0:
            return assignment, uc
        rng = random.Random(12345)
        uc = run_minconflicts(assignment, sat, uc, rng, budget)
        return assignment, uc
=======
    def polish(assignment, budget):
        # high-precision deterministic-ish min-conflicts polish
        sat_count, sat, uc = build_state(assignment)
        if uc == 0:
            return assignment, uc
        rng = random.Random(12345)
        uc = run_minconflicts(assignment, sat_count, sat, uc, rng, budget)
        return assignment, uc
>>>>>>> REPLACE

