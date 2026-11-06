            else:
                # Forced delete: pick non-output
                for u in in_fast:
                    if u not in actual_outputs:
                        operations.append(['delete', u])
                        in_fast.discard(u)
                        break
                else:
                    # All are outputs...
            