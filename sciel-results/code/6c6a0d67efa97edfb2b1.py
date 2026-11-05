            success = False
            # Try removing b
            S.remove(b)
            if not get_conflicts(a, S):
                S.add(a)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                success = True
            else:
                S.add(b)
            
            if not success:
                # Try removing c
                S.remove(c)
                if not get_conflicts(a, S):
                    S.add(a)
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    success = True
                else:
                    S.add(c)

            if not success:
                # Try removing both
                S.remove(b)
                S.remove(c)
                if not get_conflicts(a, S):
                    S.add(a)
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                else:
                    S.add(b)
                    S.add(c)
