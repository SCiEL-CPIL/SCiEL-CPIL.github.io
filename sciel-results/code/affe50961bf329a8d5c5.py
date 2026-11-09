BronKerbosch(R, P, X):
  if P and X are both empty:
    report R as a maximal clique
  for each v in P \ N(u):
    BronKerbosch(R ∪ {v}, P ∩ N(v), X ∩ N(v))
    P := P \ {v}
    X := X ∪ {v}
