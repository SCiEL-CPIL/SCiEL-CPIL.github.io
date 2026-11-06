        if len(covered) < 30:
            best_x = None
            best_score = -1
            current_max = max(S)
            
            # We collect candidates and score them
            # Score components:
            # 1. Is it inside current diameter? (Bonus)
            # 2. How many new diffs does it cover? (Primary)
            # 3. Is it small? (Secondary)
            
            # To implement the "Inside first" logic effectively:
            # We can just iterate and keep track of the best candidate.
            # If a candidate is inside, we give it a huge advantage in coverage count?
            # Or simply:
            # Pass 1: Look for candidates x <= current_max that cover > 0 new diffs.
            #   Pick the one with max new_covered (tie: min x).
            # Pass 2: If none found, look for candidates x > current_max.
            #   Pick the one with max new_covered (tie: min x).
            
            best_inside_x = None
            best_inside_covered = 0
            best_outside_x = None
            best_outside_covered = 0
            
            for x in range(1, max_x + 1):
                if x in S:
                    continue
                
                # Compute new diffs and check validity
                new_diffs = []
                valid = True
                for s in S:
                    d = abs(x - s)
                    if d == 0: continue
                    if d in existing_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                new_covered = 0
                for d in new_diffs:
                    if 1 <= d <= 30 and d not in covered:
                        new_covered += 1
                
                if new_covered == 0:
                    continue # No point adding if it doesn't help cover 1..30
                
                if x <= current_max:
                    if new_covered > best_inside_covered:
                        best_inside_covered = new_covered
                        best_inside_x = x
                else:
                    if new_covered > best_outside_covered:
                        best_outside_covered = new_covered
                        best_outside_x = x
            
            if best_inside_x is not None:
                best_x = best_inside_x
            elif best_outside_x is not None:
                best_x = best_outside_x
            else:
                # No candidate covers new diffs?
                # This shouldn't happen if we haven't covered all 30.
                # But if it does, we might need to add a "filler" that doesn't cover new diffs
                # but enables future ones? Unlikely for this specific problem structure.
                # If we are stuck, we extend max_x or break.
                # Actually, if no candidate covers a *new* diff, we are stuck in a local optimum
                # where all remaining diffs are blocked by current Sidon constraints.
                # This is rare for small N.
                max_x *= 2
                continue

            S.append(best_x)
            S.sort()
            for s in S[:-1]:
                d = abs(best_x - s)
                existing_diffs.add(d)
                if 1 <= d <= 30:
                    covered.add(d)
