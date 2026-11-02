if _verify_decomposition(all_U, all_V, all_W, n, m, p):
    if rank < schoolbook_rank:
        return {"U": all_U, "V": all_V, "W": all_W}
