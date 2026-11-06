    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            # try all moves for point i
            # if a move improves, take it, set improved=True, and break out of move loops
            # then restart the i loop? Or continue to next i?
            # Usually, if you move point i, the best move for i+1 might change.
            # So restarting the i loop (or just continuing to next i in a greedy pass) is common.
            # Greedy pass:
            for dx in ...:
                for dy in ...:
                    if better:
                        update
                        improved = True
                        break # break dy
                else:
                    continue
                break # break dx
        # After one pass over all i, if improved, loop again.
