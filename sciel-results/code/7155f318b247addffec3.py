    for a in included_set:
        if included_set & ap_partners[v][a]:
            return False
    return True
    