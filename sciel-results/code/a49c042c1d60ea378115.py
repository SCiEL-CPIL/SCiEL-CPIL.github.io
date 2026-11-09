<<<<<<< SEARCH
        if best_unsat > 0 and best_unsat <= 20:
            polished, _, _ = polish(best_assignment, best_clause_sat, deadline - 2.0)
            final_unsat = int(np.sum(best_clause_sat == 0))
            if final_unsat == 0:
                return best_assignment.tolist()
            # Try polish on current best too
            if best_unsat < unsat_count:
                best_assignment = polished
=======
        if best_unsat > 0 and best_unsat <= 30:
            polished, _, _ = polish(best_assignment, best_clause_sat, deadline - 2.0)
            final_unsat = int(np.sum(best_clause_sat == 0))
            if final_unsat == 0:
                return best_assignment.tolist()
            # Try polish on current best too
            if best_unsat < unsat_count:
                best_assignment = polished
>>>>>>> REPLACE
