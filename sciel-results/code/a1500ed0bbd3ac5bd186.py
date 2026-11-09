def update_count(triple, delta):
    for m in supersets_of_triple[triple]:
        count[m] += delta
